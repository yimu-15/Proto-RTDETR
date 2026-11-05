# Known Limitations

These limitations are disclosed in the submitted paper (Section 5) and are reproduced here for the released code.

1. **Mask→box conversion does not filter tiny noise fragments.** Connected-component labeling can produce a few single-pixel/speck boxes. They are statistically negligible (VisA has no genuine small objects, so small/medium AP is `-1.0`), but a size/area filter would further clean annotations.

2. **External SOTA baselines were not re-run locally.** The comparison table is limited to the authors' own configurations; external baselines are cited from public literature.

3. **Image-level pseudo-labels are coarse.** A defect image may contain mostly normal pixels, yet all its sampled pixels are pushed to the defect prototype. A refinement (propagating box-mask hints so only in-box pixels are "defect") is future work.

4. **Limited statistical power.** The validation set has only 239 instances; differences within ±0.03 mAP are within noise.

5. **FP32 requirement.** AMP must be disabled (`use_amp: False`) because RT-DETRv2's anchor sentinel (`torch.inf`) can produce NaN under FP16. This is a training-protocol constraint, not a model change.

6. **No EMA / query-level prototype.** The released `ProtoContrastHead` uses two **learnable (`nn.Parameter`) prototypes** optimized by backpropagation, with **encoder stride-16** features and **image-level** pseudo-labels. It does **not** implement an EMA prototype, Decoder-Query feature source, or query-level prototype learning. Any text elsewhere describing those does not match this code.

7. **FPS / parameter / VRAM figures** come from `benchmark_infer.py` on the authors' single-GPU machine and are not independently re-verified in this release.

8. **Reproducibility cost.** Each 16-epoch run is ~9.5 h on a single 8 GB GPU; the full paper table requires multiple such runs.

9. **Machine-specific paths in configs.** `configs/dataset/visa_detection.yml` and the `tuning:` field of each experiment config contain the authors' absolute paths (`D:/Proto-RTDETR/...`). You must edit these to your own data/weight locations (see `docs/reproducibility.md` §0).
