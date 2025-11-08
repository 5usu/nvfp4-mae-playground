# NVFP4 MAE Playground

Minimal-yet-practical sandbox for experimenting with **Masked Autoencoders (MAE)** under NVIDIA’s NVFP4 quantization recipes. The repo started as a pure PyTorch demo (`asmae_minimal.py`) and now supports richer data pipelines, multi-epoch runs, optional `torch.compile`, mixed precision, and automatic reconstruction dumps—without leaving the comfort of a single Python script.

---

## Highlights

- **Dataset-aware transforms** – CIFAR-10 & CIFAR-100 with normalization, color jitter, optional RandAugment.
- **Long-run friendly** – CLI flags for epochs, mask ratio, batch size, AMP, compilation, and deterministic seeds.
- **Autonomous logging** – Saves the final reconstruction grid (input | visible | reconstruction) to `runs/recons/` after every training run.
- **NVFP4 inspiration** – Mirrors the defensive coding style from NVIDIA’s NVFP4 tensor stack (contiguity checks, loss guards, mask sanity).

---

## Quick Start

```bash
git clone https://github.com/5usu/nvfp4-mae-playground.git
cd nvfp4-mae-playground
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision wandb
```

The script auto-downloads CIFAR datasets into `data/` (ignored by git).  
W&B is optional; enable it with `--use-wandb` (API key required).

---

## Training the Minimal MAE

### GPU (FP16/bfloat16 + `torch.compile`)
```bash
python asmae_minimal.py \
  --dataset cifar100 \
  --epochs 150 \
  --batch-size 128 \
  --mask-ratio 0.6 \
  --randaugment \
  --seed 42
```

### CPU-friendly pass
```bash
python asmae_minimal.py \
  --device cpu \
  --epochs 50 \
  --img-size 96 \
  --batch-size 32 \
  --no-amp \
  --no-compile
```

### CLI Cheatsheet
```
--dataset {cifar10,cifar100}     Dataset selector (default: cifar10)
--epochs INT                     Number of epochs (default: 50)
--mask-ratio FLOAT               Override MAE mask ratio
--randaugment                    Turn on torchvision.RandAugment (if available)
--amp / --no-amp                 Force-enable or disable AMP
--compile / --no-compile         Toggle torch.compile use
--seed INT                       Deterministic run (seeds PyTorch + CUDA)
```

All options run through `_parse_args()`; run `python asmae_minimal.py --help` for the complete list.

---

## Outputs

- **Checkpoints & recon grids** live under `runs/` (`runs/recons/final_recon_epXXXX.png` + `.pt` bundle).
- **Datasets** sit in `data/` and are ignored by git to avoid 300 MB pushes.

Example final image grid (saved automatically):
```
┌─────────────┬──────────────┬────────────────┐
│ original    │ visible-only │ reconstruction │
└─────────────┴──────────────┴────────────────┘
```

---

## FP4 vs FP8: Why Go Leaner?

| Aspect                  | FP8 (E4M3 / E5M2)                              | FP4 (NVFP4: float4 e2m1)                           |
|-------------------------|------------------------------------------------|---------------------------------------------------|
| Bit-width               | 8 bits                                         | 4 bits                                            |
| Mantissa bits           | 3–4                                            | 1                                                 |
| Memory & bandwidth      | Baseline                                       | ~½ vs FP8 → larger batches, faster transfer       |
| Scale management        | Per-tensor / per-channel (often manual)        | Mandatory block-wise (NVFP4BlockScaling)          |
| Random Hadamard Transform | Optional                                      | Strongly recommended (limits quantization bias)   |
| Accuracy impact         | Minimal with tuned scaling                     | Requires careful recipes but close after tuning   |
| Hardware alignment      | CUDA core friendly                             | Matches NVIDIA Transformer Engine roadmaps        |

### Why the Shift?
- **Headroom**: Cutting activations/weights to FP4 slashed memory pressure enough to fit deeper ViT blocks and larger batches on the same GPUs.
- **Throughput**: Fewer bytes move across NVLink/PCIe; kernels reach higher occupancy, especially with TE-backed quantization.
- **Predictable recipes**: Transformer Engine already exposes FP4 quantizers, scale tensors, and Random Hadamard transforms—so the migration keeps the code lean while reusing proven kernels.
- **Comparable accuracy**: Once block scaling + stochastic rounding are in place, reconstruction quality stayed close to FP8 while delivering better perf/watt.

In short: FP4 demands disciplined scaling but rewards you with a leaner training footprint and future-ready compatibility with NVIDIA’s quantization roadmap.

---


