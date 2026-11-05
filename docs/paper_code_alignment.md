# Paper ↔ Code Alignment

This document maps each claim in the submitted paper *Proto-Contrast Learning for Industrial Defect Detection with RT-DETRv2* to concrete code evidence. The authoritative code version is the self-contained package under `src/`, `configs/`, and `tools/` in this repository (which corresponds to the original submission package; it is a *simpler, cleaner* version than the internal working directory that accumulated later unrelated diagnostics).

**Headline conclusion: the paper and the released code are consistent.** In particular, the paper describes *encoder stride-16 features + random 256-token sampling + two learnable (`nn.Parameter`) prototypes + InfoNCE/softmax-CE*, and the code implements exactly that. It does **not** use an EMA prototype, a Decoder-Query feature source, or a Hungarian matcher inside the head.

---

## 1. ProtoContrastHead

| Paper claim | Code evidence |
|---|---|
| "learns two class prototypes" | [`src/nn/proto_contrast_head.py`](src/nn/proto_contrast_head.py) — `self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))` (`num_classes=2`) |
| "initialized with Xavier" | `nn.init.xavier_uniform_(self.prototypes)` |
| "small projection MLP (2-layer 256→256→128 with LayerNorm + ReLU)" | `self.proj = nn.Sequential(Linear(256,256), LayerNorm(256), ReLU, Linear(256,128))` |
| "features and prototypes are L2-normalized" | `F.normalize(proj_feat, dim=-1)` and `F.normalize(proto_proj, dim=-1)` |
| "InfoNCE / softmax cross-entropy over two prototypes" | `_info_nce_loss` computes `sim = proj_feat @ proto_proj.T / τ` then `F.cross_entropy(sim, labels)` |
| "temperature τ = 0.07" | constructor default `temperature: float = 0.07`; confirmed by `configs/proto_rtdetrv2_r50vd_visa*.yml` (`temperature: 0.07`) |
| "optimized end-to-end by backpropagation (no EMA)" | prototypes are `nn.Parameter` (trainable); the module docstring states "原型通过反向传播端到端优化（与论文一致，无 EMA）". There is **no** EMA/momentum update for prototypes anywhere in the repo. |
| "loss weight applied once" | `result["loss_contrast"] = loss * self.loss_weight` (applied inside the head, single multiplication) |

## 2. Feature source and labels

| Paper claim | Code evidence |
|---|---|
| "mid-level (stride-16) feature map" | [`src/zoo/rtdetr/rtdetr.py`](src/zoo/rtdetr/rtdetr.py) — `mid_feat = x[len(x) // 2]` (the middle scale of the HybridEncoder output, stride 16 at 640×640 → `(B,256,40,40)`) |
| "randomly sample at most S=256 spatial positions" | `sample_per_img = min(256, H*W)`; `indices = torch.randperm(H*W)[:sample_per_img]` |
| "image-level pseudo-labels (defect if image contains ≥1 GT box)" | `has_defect = len(targets[b].get("labels", [])) > 0`; `lbl = 0 if has_defect else 1` |
| "all sampled pixels of an image share the same label" | `torch.full((sample_per_img,), lbl, ...)` |

The feature source is the **encoder** intermediate feature, **not** a decoder query. No decoder-query tensor is passed to `ProtoContrastHead`.

## 3. Hungarian matching

The paper mentions Hungarian matching only as background (DETR / RT-DETR's base detector, Section 2.2 / 3.1). The Hungarian matcher is part of the **upstream RT-DETRv2 base detector** ([`src/zoo/rtdetr/matcher.py`](src/zoo/rtdetr/matcher.py), [`src/zoo/rtdetr/rtdetrv2_criterion.py`](src/zoo/rtdetr/rtdetrv2_criterion.py)) and is **not** part of `ProtoContrastHead`. The head consumes `(features, targets)` only; it performs no matching.

## 4. Training-only execution

| Paper claim | Code evidence |
|---|---|
| "executed only during training" | `rtdetr.py` gate: `if self.proto_head is not None and self.training and targets is not None:` |
| "no additional computational cost at inference" | the head forward is skipped when `not self.training` or `targets is None`; inference path is `deploy() → eval()`. |

## 5. Data augmentation / Mosaic

| Paper claim | Code evidence |
|---|---|
| default augmentation pipeline (PhotometricDistort, RandomZoomOut, RandomIoUCrop, RandomHorizontalFlip, Resize 640, optional Mosaic) | `configs/rtdetrv2/include/dataloader.yml` (upstream); the paper's no-Mosaic configs override `ops` to remove Mosaic |
| Mosaic removed in final config | `configs/proto_rtdetrv2_r50vd_visa_abl2_lw1.yml` and `..._abl_no_mosaic.yml` |

## 6. VisA conversion & split

| Paper claim | Code evidence |
|---|---|
| mask → tight box via connected components | [`tools/visa_csv2coco.py`](tools/visa_csv2coco.py) (`scipy.ndimage.label`) |
| 12 object types → single `defect` class | `category_id=1` in the conversion script |
| 80/20 random split (non-official) | `RANDOM_SEED=42` in the conversion script |

> Note: the split is the **authors' own 80/20 random split (seed 42)**, not VisA's official 1-class/2-class splits. See `docs/dataset_protocol.md`.

## 7. Numbers verification

All paper numbers were extracted from `output/<run>/log.txt` (per-epoch `test_coco_eval_bbox`). The released repo does **not** ship these logs/checkpoints (they are large); the mapping from config → best mAP is recorded in [`results/paper_results.md`](../results/paper_results.md) and is traceable to the original `output/` directory on the authors' machine.

## 8. Deliberately not in the released repo

- No EMA prototype, no query-level prototype learning — **the paper never claims these**, and the code does not implement them.
- No later, unverified research (e.g., the Box-aware MIL prototype variants, spatial-purity diagnostics, F2/F3 arms) is included.

## 9. UNVERIFIED items

- External SOTA baselines (cited from literature) were **not re-run locally**; the paper states this limitation explicitly (Section 5, Limitation 2).
- FPS/parameter/VRAM numbers come from `benchmark_infer.py` on the authors' single-GPU machine; they are not independently re-verified here.
