# Dataset Protocol

This repository targets **VisA** for the submitted paper. The VisA images and pretrained weights are **not** redistributed.

## 1. Data acquisition

Download VisA from the official source:

- **Direct download:** `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`
- **Official repo / docs:** <https://github.com/amazon-science/spot-diff>
- **AWS Open Data registry:** <https://registry.opendata.aws/visa/>
- **License:** CC BY 4.0

## 2. Directory layout (expected)

```
datasets/
└── visa/
    ├── VisA/                     # raw images + masks (official VisA layout)
    │   └── data/VisA_20220922/...
    └── annotations/              # generated COCO JSON (by tools/visa_csv2coco.py)
        ├── train.json
        └── test.json
```

## 3. Split actually used in the paper

The paper uses the authors' own **80/20 random split (seed 42, stratified by object)**, produced by `tools/visa_csv2coco.py`. It is **not** VisA's official 1-class / 2-class split.

| Split | Images | Boxes |
|---|---|---|
| Train | 8,654 | 961 |
| Val (referred to as "test.json") | 2,167 | 239 |

> The string `test.json` in the conversion script is the **validation** split used for model selection in the paper (no separate held-out test was used for method development). See `docs/reproducibility.md`.

## 4. Mask → Box conversion

`tools/visa_csv2coco.py`:

- Maps the 12 object types to a single `defect` class (`category_id=1`).
- Converts each defect mask to tight bounding boxes via connected-component labeling (`scipy.ndimage.label`); a single defect may yield several boxes.
- Binarizes masks at gray level `> 0` (VisA encodes masks as 0/5).
- Tolerant CSV-header parsing.
- Optional `--cc` connected-component check.

Run:

```powershell
python tools/visa_csv2coco.py --help
python tools/visa_csv2coco.py
```

## 5. Data / label integrity check

`check_dataset.py` validates that the produced COCO annotations are consistent (image/box counts, bbox validity). Run after conversion:

```powershell
python check_dataset.py
```

## 6. Note on a later "VisA V2" line

A **later, post-submission** reconstruction ("VisA V2 Mask-BBox", with different split counts and a separate `configs/visa_v2/` experiment line) exists in the authors' internal working directory. It is **not** part of the submitted paper and is **intentionally excluded** from this repository. Do not mix the two protocols' numbers.
