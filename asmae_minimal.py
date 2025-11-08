# asmae_minimal.py
import argparse
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torch import amp

DATASET_REGISTRY = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "cls": datasets.CIFAR100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}

# ---- Patchify / unpatchify helpers ----
def patchify(imgs, patch_size: int):  # imgs: [B, C, H, W]
    p = patch_size
    if not imgs.is_contiguous():
        imgs = imgs.contiguous()
    B, C, H, W = imgs.shape
    assert H % p == 0 and W % p == 0
    h = H // p
    w = W // p
    x = imgs.reshape(B, C, h, p, w, p).permute(0,2,4,3,5,1).reshape(B, h*w, p*p*C)
    return x  # [B, N, patch_dim]

def unpatchify(patches, patch_size: int, C: int, H: int, W: int):
    B, N, D = patches.shape
    p = patch_size
    h, w = H // p, W // p
    if not patches.is_contiguous():
        patches = patches.contiguous()
    x = patches.reshape(B, h, w, p, p, C).permute(0,5,1,3,2,4).reshape(B, C, H, W)
    return x

# ---- Positional embeddings ----
class PosEmbed(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        pe = torch.zeros(1, num_patches, dim)
        # simple fixed sinusoidal pe
        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---- Tiny ViT-style blocks ----
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim*mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x) + h
        return x

# ---- Masking (random indices, return visible+mask info) ----
def random_masking(x, mask_ratio: float):
    B, N, D = x.shape
    if not 0 < mask_ratio < 1:
        raise ValueError(f"mask_ratio must be in (0,1), got {mask_ratio}")
    len_keep = int(N * (1 - mask_ratio))
    if len_keep <= 0:
        len_keep = 1
    noise = torch.rand(B, N, device=x.device)  # small→keep
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1,1,D))
    # binary mask: 0 keep, 1 remove
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_keep, mask, ids_restore

# ---- AS-MAE core ----
@dataclass
class ASMAEConfig:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    enc_dim: int = 384
    enc_depth: int = 6
    enc_heads: int = 6
    dec_dim: int = 192
    dec_depth: int = 4
    dec_heads: int = 6
    mask_ratio: float = 0.75

class ASMAE(nn.Module):
    def __init__(self, cfg: ASMAEConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        self.num_patches = (cfg.img_size // p) ** 2
        patch_dim = p*p*cfg.in_chans

        # patch embed (linear)
        self.proj = nn.Linear(patch_dim, cfg.enc_dim)
        self.pos_enc = PosEmbed(self.num_patches, cfg.enc_dim)

        # encoder (heavy)
        self.enc_blocks = nn.ModuleList([Block(cfg.enc_dim, cfg.enc_heads) for _ in range(cfg.enc_depth)])
        self.enc_norm = nn.LayerNorm(cfg.enc_dim)

        # decoder (light)
        self.dec_embed = nn.Linear(cfg.enc_dim, cfg.dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        self.pos_dec = PosEmbed(self.num_patches, cfg.dec_dim)
        self.dec_blocks = nn.ModuleList([Block(cfg.dec_dim, cfg.dec_heads, mlp_ratio=2.0) for _ in range(cfg.dec_depth)])
        self.dec_norm = nn.LayerNorm(cfg.dec_dim)
        self.pred = nn.Linear(cfg.dec_dim, patch_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, imgs):
        B, C, H, W = imgs.shape
        patches = patchify(imgs, self.cfg.patch_size)                   # [B, N, p^2*C]
        x = self.proj(patches)                                          # [B, N, enc_dim]
        x = self.pos_enc(x)

        x_vis, mask, ids_restore = random_masking(x, self.cfg.mask_ratio)

        # encoder on visible patches
        for blk in self.enc_blocks:
            x_vis = blk(x_vis)
        latents = self.enc_norm(x_vis)                                  # [B, N_vis, enc_dim]

        # decoder: re-insert mask tokens
        x = self.dec_embed(latents)
        num_masked = self.num_patches - x.size(1)
        if num_masked > 0:
            mask_tokens = self.mask_token.expand(B, num_masked, -1)
            x_ = torch.cat([x, mask_tokens], dim=1)                     # [B, N, dec_dim] (shuffled)
        else:
            x_ = x
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(-1))
        x_ = torch.gather(x_, 1, ids_restore_expanded)
        x_ = self.pos_dec(x_)
        for blk in self.dec_blocks:
            x_ = blk(x_)
        x_ = self.dec_norm(x_)
        pred = self.pred(x_)                                            # [B, N, patch_dim]

        # L2 only on masked patches
        target = patches
        loss = (pred - target) ** 2
        denom = mask.sum().clamp_min(1.0)
        loss = (loss.mean(dim=-1) * mask).sum() / denom
        return loss, pred, mask, latents

# ---- Quick training demo on CIFAR-style datasets ----
def _build_transforms(dataset_name: str, img_size: int, use_randaugment: bool):
    entry = DATASET_REGISTRY[dataset_name]
    normalize = transforms.Normalize(mean=entry["mean"], std=entry["std"])
    tfm_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    ]
    if use_randaugment and hasattr(transforms, "RandAugment"):
        tfm_list.append(transforms.RandAugment())
    tfm_list.extend([
        transforms.ToTensor(),
        normalize,
    ])
    return transforms.Compose(tfm_list)


def _build_dataset(dataset_name: str, data_root: str, train: bool, transform):
    entry = DATASET_REGISTRY[dataset_name]
    dataset_cls = entry["cls"]
    if dataset_name.startswith("cifar"):
        return dataset_cls(root=data_root, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset {dataset_name}")


def _denorm(imgs: torch.Tensor, mean, std):
    device = imgs.device
    dtype = imgs.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return (imgs * std_t) + mean_t


def demo_train(
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda",
    img_size: int = 224,
    dataset: str = "cifar10",
    data_root: str = "./data",
    mask_ratio: Optional[float] = None,
    use_randaugment: bool = False,
    num_workers: Optional[int] = None,
    use_amp: bool = True,
    compile_model: bool = False,
    cfg: Optional[ASMAEConfig] = None,
):
    dataset = dataset.lower()
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"dataset must be one of {list(DATASET_REGISTRY)}, got {dataset}")
    device_obj = torch.device(device)
    is_cuda = device_obj.type == "cuda"
    if is_cuda:
        torch.backends.cudnn.benchmark = True
    cfg = cfg or ASMAEConfig(img_size=img_size)
    cfg.img_size = img_size
    if mask_ratio is not None:
        cfg.mask_ratio = mask_ratio

    tfm = _build_transforms(dataset, cfg.img_size, use_randaugment)
    ds = _build_dataset(dataset, data_root, train=True, transform=tfm)
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=is_cuda,
    )
    if num_workers > 0 and is_cuda:
        loader_kwargs["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    model = ASMAE(cfg).to(device_obj)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.05)
    scaler = amp.GradScaler(enabled=is_cuda and use_amp)

    amp_dtype = None
    if is_cuda and scaler.is_enabled():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model.train()
    for ep in range(epochs):
        running = 0.0
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device_obj, non_blocking=is_cuda)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                autocast_cm = amp.autocast("cuda", dtype=amp_dtype)
            else:
                autocast_cm = nullcontext()
            with autocast_cm:
                loss, _, _, _ = model(imgs)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            running += loss.item()
            if (i+1) % 100 == 0:
                if is_cuda:
                    torch.cuda.synchronize()
                print(f"epoch {ep+1} iter {i+1}: L_recon={running/100:.4f}")
                running = 0.0

    # Save final reconstruction grid
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device_obj)[:8]
        loss, pred, mask, _ = model(imgs)
        C = cfg.in_chans
        p = cfg.patch_size
        rec = unpatchify(pred, p, C, cfg.img_size, cfg.img_size)
        patches_in = patchify(imgs, p)
        vis_mask = (1 - mask).unsqueeze(-1)
        vis_patches = patches_in * vis_mask
        vis_img = unpatchify(vis_patches, p, C, cfg.img_size, cfg.img_size)

        entry = DATASET_REGISTRY[dataset]
        imgs_denorm = _denorm(imgs.detach(), entry["mean"], entry["std"]).cpu().clamp(0, 1)
        vis_denorm = _denorm(vis_img.detach(), entry["mean"], entry["std"]).cpu().clamp(0, 1)
        rec_denorm = _denorm(rec.detach(), entry["mean"], entry["std"]).cpu().clamp(0, 1)

        grid = torch.cat([imgs_denorm, vis_denorm, rec_denorm], dim=0)
        grid = make_grid(grid, nrow=imgs_denorm.size(0))
        os.makedirs("runs/recons", exist_ok=True)
        out_png = os.path.join("runs/recons", f"final_recon_ep{epochs:04d}.png")
        save_image(grid, out_png)
        torch.save(
            {
                "imgs": imgs_denorm,
                "visible": vis_denorm,
                "recon": rec_denorm,
                "loss": loss.item(),
            },
            os.path.join("runs/recons", f"final_recon_ep{epochs:04d}.pt"),
        )
        print(f"[final] saved reconstruction snapshot to {out_png}")

def _parse_args():
    parser = argparse.ArgumentParser(description="Train the minimal ASMAE model on CIFAR-style datasets.")
    parser.add_argument("--dataset", type=str, choices=sorted(DATASET_REGISTRY.keys()), default="cifar10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to 128 on CUDA, 32 on CPU.")
    parser.add_argument("--lr", type=float, default=None, help="Defaults to 1e-4 on CUDA, 3e-4 on CPU.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detected if omitted).")
    parser.add_argument("--img-size", type=int, default=None, help="Defaults to 224 on CUDA, 96 on CPU.")
    parser.add_argument("--mask-ratio", type=float, default=None, help="Override ASMAEConfig.mask_ratio.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--randaugment", action="store_true", help="Apply RandAugment in the data pipeline.")
    parser.add_argument("--amp", action="store_true", help="Force enable AMP.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even if CUDA is available.")
    parser.add_argument("--compile", action="store_true", help="Force torch.compile.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _maybe_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = _parse_args()
    if args.amp and args.no_amp:
        raise ValueError("Cannot pass both --amp and --no-amp.")
    if args.compile and args.no_compile:
        raise ValueError("Cannot pass both --compile and --no-compile.")

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = (args.device or auto_device).lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    is_cuda = device == "cuda"

    use_amp = args.amp if args.amp else is_cuda
    if args.no_amp:
        use_amp = False

    compile_model = args.compile if args.compile else is_cuda
    if args.no_compile:
        compile_model = False

    batch_size = args.batch_size if args.batch_size is not None else (128 if is_cuda else 32)
    lr = args.lr if args.lr is not None else (1e-4 if is_cuda else 3e-4)
    img_size = args.img_size if args.img_size is not None else (224 if is_cuda else 96)

    _maybe_seed(args.seed)

    demo_train(
        epochs=args.epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        img_size=img_size,
        dataset=args.dataset,
        data_root=args.data_root,
        mask_ratio=args.mask_ratio,
        use_randaugment=args.randaugment,
        num_workers=args.num_workers,
        use_amp=use_amp,
        compile_model=compile_model,
    )


if __name__ == "__main__":
    main()

