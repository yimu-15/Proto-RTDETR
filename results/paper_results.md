# Paper Results

All numbers below are reproduced from the submitted paper and are traceable to the authors' original `output/<run>/log.txt` (per-epoch `test_coco_eval_bbox`) and `best.pth` checkpoints. They were **not** independently re-run as part of this release; full 16-epoch reproduction is left to the user (see `docs/reproducibility.md`).

## Table 1 — VisA validation (16 epochs, RT-DETRv2 R50vd, 640×640)

| Method | mAP@[.5:.95] | AP@.50 | AP@.75 | AR@100 | Params (trainable) |
|---|---|---|---|---|---|
| YOLOv8m (reference, default aug.) | 0.077 | 0.145 | 0.090 | 0.958 | 25.84 M |
| YOLOv8m (reference, no Mosaic) | 0.057 | 0.114 | 0.030 | 0.891 | 25.84 M |
| RT-DETRv2 R50vd (default aug.) | 0.255 | 0.401 | 0.288 | 0.772 | 42.70 M |
| + ProtoContrastHead (λ=0.5) | 0.329 | 0.531 | 0.351 | 0.804 | 42.80 M |
| RT-DETRv2 R50vd (no Mosaic) | 0.884 | 0.891 | 0.882 | 0.994 | 42.70 M |
| + ProtoContrastHead (λ=0.5) | 0.857 | 0.860 | 0.855 | 0.997 | 42.80 M |
| + ProtoContrastHead (λ=1.0) | **0.890** | 0.898 | 0.891 | 0.993 | 42.80 M |

## Ablation A — head under default (Mosaic) pipeline

| ProtoContrastHead | λ | mAP | Δ |
|---|---|---|---|
| ✗ | — | 0.255 | — |
| ✓ | 0.5 | 0.329 | +0.074 |
| ✓ | 1.0 | 0.333 | +0.078 |

## Ablation B — head under no-Mosaic pipeline (final setting)

| ProtoContrastHead | λ | mAP | best epoch |
|---|---|---|---|
| ✗ | — | 0.884 | 13 |
| ✓ | 0.5 | 0.857 | 15 |
| ✓ | 1.0 | 0.890 | 10 |

## Ablation C — denoising queries

| num_denoising | mAP |
|---|---|
| 30 | 0.329 |
| 100 | 0.329 |

## Ablation E — temperature sensitivity (6-epoch budget; last row = 16-epoch final)

| τ | Epochs | mAP | best epoch |
|---|---|---|---|
| 0.02 | 6 | 0.352 | 5 |
| 0.07 | 6 | 0.350 | 5 |
| 0.10 | 6 | 0.284 | 3 |
| 0.30 | 6 | 0.291 | 2 |
| 0.07 | 16 | 0.890 | 10 |

## Section 4.6 — runtime / parameters

| Model | Trainable Params | Total Params | Peak VRAM | Inference FPS |
|---|---|---|---|---|
| RT-DETRv2 R50vd | 42.70 M | 42.73 M | 0.314 GB | 31.3 |
| + ProtoContrastHead | 42.80 M (+0.23%) | 42.83 M | 0.315 GB | 31.1 |

> Source of these figures: `benchmark_infer.py` on the authors' machine. Not independently re-verified.
