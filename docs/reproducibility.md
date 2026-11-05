# Reproducibility

## 0. Paths you must edit before training

The configs ship with the authors' machine-specific absolute paths. **Update these to your own locations** before running:

1. `configs/dataset/visa_detection.yml` — `train_dataloader.dataset.img_folder`, `ann_file`, and `val_dataloader.dataset.img_folder`, `ann_file` (point to your VisA images and the `train.json` / `test.json` produced by `tools/visa_csv2coco.py`).
2. Each `configs/proto_rtdetrv2_r50vd_visa*.yml` — the top-level `tuning:` field (point to your downloaded `rtdetrv2_r50vd_6x_coco_ema.pth`).

No other machine-specific paths exist in the code (there is no `.env`, no hardcoded credentials).

## 1. Determinism

All experiments use `--seed 42` (default in `train_proto_visa.py`). `setup_seed()` fixes `random`, `numpy`, `torch`, and CUDA, and disables cuDNN non-determinism:

```python
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Data split uses `RANDOM_SEED=42` in `tools/visa_csv2coco.py`.

## 2. Training settings

| Item | Value |
|---|---|
| Base model | RT-DETRv2 R50vd |
| Input / eval size | 640×640 |
| Pretrained weights | `rtdetrv2_r50vd_6x_coco_ema.pth` (upstream COCO) |
| Optimizer | AdamW, lr 1e-4 (decoder/encoder), backbone lr 1e-5, wd 1e-3 |
| Scheduler | MultiStepLR [11, 14], γ=0.1, 16 epochs, linear warmup 2000 steps |
| EMA | decay 0.9999, warmup 2000 steps (validation uses EMA weights) |
| Denoising | `num_denoising=30` |
| Loss weights | VFL 1, bbox 5, GIoU 2 |
| Batch size / workers | 2 / 0 (Windows) |
| AMP | off (`use_amp: False`) |
| Gradient clip | `clip_max_norm=0.1` |

## 3. Reproducing the paper tables

From the repository root:

```powershell
# Table 1 (default augmentation, with Mosaic)
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl_wo_proto.yml --seed 42   # 0.255
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa.yml --seed 42                # 0.329
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl_lw1.yml --seed 42        # 0.333

# Table 1 (no Mosaic)
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl2_wo_proto.yml --seed 42  # 0.884
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl_no_mosaic.yml --seed 42  # 0.857
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl2_lw1.yml --seed 42       # 0.890

# Ablation C (denoising=100)
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl_dn100.yml --seed 42      # 0.329

# Ablation E (temperature, 6-epoch budget)
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl2_lw1_tau002.yml --seed 42
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl2_lw1_tau007.yml --seed 42
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl2_lw1_tau01.yml  --seed 42
python train_proto_visa.py -c configs/proto_rtdetrv2_r50vd_visa_abl2_lw1_tau03.yml  --seed 42
```

## 4. Extracting metrics

Each run writes per-epoch metrics to `output/<run>/log.txt` (JSON lines). Parse `test_coco_eval_bbox` (the COCO `[mAP, AP50, AP75, ..., AR]` vector) to get the best-epoch numbers.

## 5. Figure generation

```powershell
python paper/generate_figures.py
```

Reads `output/*/log.txt` and writes figures to `paper/figures/`.

## 6. What is / is not independently re-verified here

- The released code passes import / config-load / single-batch forward & backward checks (see the minimum reproduction report in this repo's release notes).
- Full 16-epoch training was **not** re-run as part of this release; the reported numbers trace to the authors' original `output/` logs and checkpoints.
- FPS / parameter / VRAM figures come from `benchmark_infer.py` on the authors' machine.
