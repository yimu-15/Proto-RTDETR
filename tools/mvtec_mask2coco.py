# D:\Proto-RTDETR\tools\mvtec_mask2coco.py
import os, json, glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

MVTEC_ROOT = r"D:\Proto-RTDETR\datasets\mvtec_ad"
OUTPUT_DIR = r"D:\Proto-RTDETR\datasets\mvtec_ad\annotations"
K_SHOT = 5  # ← 可修改为 1, 3, 5, 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_to_bbox(mask_path):
    mask = np.array(Image.open(mask_path).convert("L")) > 128
    coords = np.where(mask)
    if len(coords[0]) == 0: return None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return [int(x_min), int(y_min), int(x_max-x_min+1), int(y_max-y_min+1)]

categories = []
cat_id = 1
all_classes = sorted([d for d in os.listdir(MVTEC_ROOT) 
                      if os.path.isdir(os.path.join(MVTEC_ROOT, d)) and d != 'annotations'])

for cls_name in all_classes:
    categories.append({"id": cat_id, "name": cls_name})
    cat_id += 1

coco_train = {"images": [], "annotations": [], "categories": categories}
coco_val = {"images": [], "annotations": [], "categories": categories}
cat_map = {c["name"]: c["id"] for c in categories}

img_id, ann_id = 1, 1

for cls_name in tqdm(all_classes, desc="Processing classes"):
    cls_dir = os.path.join(MVTEC_ROOT, cls_name)
    
    # === 训练集：仅正常样本，K-shot采样 ===
    good_imgs = sorted(glob.glob(os.path.join(cls_dir, "train/good/*.png")))
    sampled = random.sample(good_imgs, min(K_SHOT, len(good_imgs)))
    
    for img_path in sampled:
        w, h = Image.open(img_path).size
        coco_train["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})
        img_id += 1
    
    # === 验证/测试集：正常 + 异常（带bbox） ===
    test_good = sorted(glob.glob(os.path.join(cls_dir, "test/good/*.png")))
    for img_path in test_good:
        w, h = Image.open(img_path).size
        coco_val["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})
        img_id += 1
    
    # 缺陷样本
    defect_types = [d for d in os.listdir(os.path.join(cls_dir, "test")) if d != "good"]
    for defect in defect_types:
        defect_imgs = sorted(glob.glob(os.path.join(cls_dir, f"test/{defect}/*.png")))
        for img_path in defect_imgs:
            w, h = Image.open(img_path).size
            coco_val["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})
            
            mask_path = os.path.join(cls_dir, f"ground_truth/{defect}/{os.path.basename(img_path).replace('.png', '_mask.png')}")
            bbox = mask_to_bbox(mask_path) if os.path.exists(mask_path) else None
            if bbox:
                coco_val["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_map[cls_name],
                    "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": 0
                })
                ann_id += 1
            img_id += 1

for name, data in [("mvtec_train_k{}.json".format(K_SHOT), coco_train), 
                   ("mvtec_val.json", coco_val)]:
    with open(os.path.join(OUTPUT_DIR, name), "w") as f:
        json.dump(data, f)
    print(f"✅ {name}: {len(data['images'])} imgs, {len(data['annotations'])} anns")