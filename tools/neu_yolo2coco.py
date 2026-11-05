# D:\Proto-RTDETR\tools\neu_yolo2coco.py
import os
import json
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = r"D:\Proto-RTDETR\datasets\NEU-DET"
OUT_ANNO = os.path.join(ROOT, "annotations")
os.makedirs(OUT_ANNO, exist_ok=True)

with open(os.path.join(ROOT, "data.yaml"), "r", encoding="utf-8") as f:
    cls_list = yaml.safe_load(f)["names"]
coco_categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(cls_list)]

def get_real_path(full_old_path: str):
    seg = full_old_path.split("NEU-DET")[-1]
    real_full = os.path.join(ROOT, seg.strip("\\/"))
    return real_full

def convert(txt_name: str, json_name: str):
    coco_data = {"images": [], "annotations": [], "categories": coco_categories}
    img_id = 1
    ann_id = 1
    lost = 0

    txt_path = os.path.join(ROOT, txt_name)
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [s.strip() for s in f if s.strip()]

    print(f"\n======== {txt_name} 路径校验 ========")
    for line in lines[:5]:
        new_p = get_real_path(line)
        ok = os.path.exists(new_p)
        print(f"原路径：{line}")
        print(f"真实路径：{new_p} | 存在：{ok}\n")

    for line in tqdm(lines, desc=json_name):
        img_abs = get_real_path(line)
        if not os.path.exists(img_abs):
            lost += 1
            continue

        label_abs = img_abs.replace("images", "labels")
        label_abs = os.path.splitext(label_abs)[0] + ".txt"

        w, h = Image.open(img_abs).size
        rel_path = os.path.relpath(img_abs, ROOT)
        coco_data["images"].append({
            "id": img_id,
            "file_name": rel_path,
            "width": w,
            "height": h
        })

        # 容错读取标签
        if os.path.exists(label_abs):
            with open(label_abs, "r", encoding="utf-8") as f:
                for lab in f.readlines():
                    lab = lab.strip()
                    if not lab:
                        continue
                    parts = lab.split()
                    # 必须满足标准yolo5个字段才解析
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, nw, nh = map(float, parts[1:5])
                    x = (cx - nw / 2) * w
                    y = (cy - nh / 2) * h
                    bw = nw * w
                    bh = nh * h
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid + 1,
                        "bbox": [round(x), round(y), round(bw), round(bh)],
                        "area": round(bw * bh),
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1

    save_file = os.path.join(OUT_ANNO, json_name)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)
    print(f"【完成】图片:{len(coco_data['images'])} 标注:{len(coco_data['annotations'])} 丢失:{lost}\n")

if __name__ == "__main__":
    test_img = r"D:\Proto-RTDETR\datasets\NEU-DET\train\images\crazing_10.jpg"
    print("测试图片存在：", os.path.exists(test_img))
    convert("train.txt", "neu_det_train.json")
    convert("val.txt", "neu_det_val.json")