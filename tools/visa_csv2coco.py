"""
VisA CSV → COCO JSON 转换脚本
适配 VisA 真实目录结构：每个 object 下有独立的 image_anno.csv
- 12 个 object，每个 object 的缺陷统一归为 "defect" 类
- 从 mask PNG 自动提取 bounding box（取 mask 外接矩形）
- 输出 train.json 和 test.json（按 2:8 比例划分）
"""

import os
import json
import csv
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from scipy import ndimage

# ============== 配置 ==============
VISA_ROOT = Path(r"D:\Proto-RTDETR\datasets\visa\VisA\data\VisA_20220922")
OUTPUT_DIR = Path(r"D:\Proto-RTDETR\datasets\visa\annotations")
TRAIN_RATIO = 0.8  # 80% 训练, 20% 测试
RANDOM_SEED = 42

# 12 个 object 名称
OBJECTS = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]

# 类别定义：每个 object 有 "normal" 和 "defect" 两类
# 但为了通用 few-shot 实验，我们统一使用 "defect" 作为唯一的缺陷类
CATEGORIES = [
    {"id": 1, "name": "defect", "supercategory": "defect"},
]

def parse_args():
    """命令行参数（默认值保持与历史实验一致，确保 GT 可复现）。"""
    parser = argparse.ArgumentParser(description="VisA CSV → COCO JSON 转换")
    parser.add_argument("--visa-root", type=str, default=str(VISA_ROOT),
                        help="VisA 数据集根目录")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="COCO JSON 输出目录")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO,
                        help="训练集比例 (默认 0.8)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="随机划分种子 (默认 42)")
    parser.add_argument("--cc", action="store_true",
                        help="连通域分析: 每个缺陷连通域生成独立 bbox。"
                             "注意: 默认关闭以保持与已发表实验相同的 GT，开启后生成的标注与论文数据不一致")
    parser.add_argument("--check", action="store_true",
                        help="审计模式: 只做一致性/连通域统计，不写任何文件")
    return parser.parse_args()


def extract_bboxes(mask_path: str, img_size: tuple, use_cc: bool = False, min_area: float = 1.0):
    """从 mask PNG 提取 bounding box 列表。

    use_cc=False: 整块 mask 取外接矩形（1 个 bbox）——与已发表实验的 GT 完全一致。
    use_cc=True : 连通域分析（scipy.ndimage.label），每个连通域 1 个 bbox——多缺陷图得到多个 bbox。

    返回: [(bbox, segmentation), ...]；读取失败或 mask 全空时返回 []。
    """
    try:
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img)
    except Exception as e:
        print(f"    ⚠️ 无法读取 mask: {mask_path} -> {e}")
        return []

    # 二值化：非零区域为缺陷（注意 VisA mask 像素值为 0/5，不是 0/255）
    binary = (mask_np > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    w_img, h_img = img_size
    results = []

    if use_cc:
        labeled, n_comp = ndimage.label(binary)
        for i in range(1, n_comp + 1):
            ys, xs = np.where(labeled == i)
            area = float(len(ys))
            if area < min_area:
                continue
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            w = float(x_max - x_min + 1)
            h = float(y_max - y_min + 1)
            bbox = [float(x_min), float(y_min), w, h]
            seg = [float(x_min), float(y_min), float(x_max + 1), float(y_min),
                   float(x_max + 1), float(y_max + 1), float(x_min), float(y_max + 1)]
            results.append((bbox, [seg]))
    else:
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        w = float(x_max - x_min + 1)
        h = float(y_max - y_min + 1)
        bbox = [float(x_min), float(y_min), w, h]
        seg = [float(x_min), float(y_min), float(x_max + 1), float(y_min),
               float(x_max + 1), float(y_max + 1), float(x_min), float(y_max + 1)]
        results.append((bbox, [seg]))

    return results


def get_image_size(img_path: str):
    """获取图片尺寸"""
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"  ⚠️ 无法读取图片: {img_path} -> {e}")
        return None


def iter_visa_rows(csv_path: Path):
    """按行读取 VisA CSV，容错被污染的表头（如 candle 首列表头被 PS 命令残片污染，
    真实列名 'image' 被拼接到末尾）。按列定位读取：精确匹配优先，后缀匹配兜底。
    yield (image_rel, label, mask_rel)；表头异常时打印错误并返回。
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        def locate(name):
            for i, h in enumerate(header):
                if h.strip() == name:
                    return i
            for i, h in enumerate(header):
                if h.strip().endswith(name):
                    return i
            return None

        c_img = locate("image")
        c_label = locate("label")
        c_mask = locate("mask")

        if c_img is None or c_label is None:
            print(f"  ❌ CSV 表头缺少 image/label 列: {csv_path} -> {header}")
            return

        expected = ("image", "label", "mask")
        if not all(h.strip() == n for h, n in zip(header, expected)):
            print(f"  ⚠️ CSV 表头与预期不符（已按列位置容错读取）: {csv_path}")
            print(f"      实际表头: {header}")

        for row in reader:
            if len(row) <= c_img or not row[c_img].strip():
                continue
            image_rel = row[c_img].strip()
            label = row[c_label].strip()
            mask_rel = row[c_mask].strip() if (c_mask is not None and len(row) > c_mask) else ""
            yield image_rel, label, mask_rel


def process_object(obj_name: str, data_root: Path, use_cc: bool = False, stats: dict = None):
    """处理单个 object 的 CSV，返回 images 和 annotations 列表。
    stats 用于累计一致性警告计数（缺 mask / 空 mask / fallback / 越界 / 多连通域）。
    """
    csv_path = data_root / obj_name / "image_anno.csv"
    
    if not csv_path.exists():
        print(f"  ❌ CSV 不存在: {csv_path}")
        return [], []
    
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    
    for image_rel, label, mask_rel in iter_visa_rows(csv_path):
        # 图片绝对路径
        img_abs = data_root / image_rel
        if not img_abs.exists():
            print(f"  ⚠️ 图片不存在: {img_abs}")
            stats["missing_image"] += 1
            continue

        # 获取图片尺寸
        img_size = get_image_size(str(img_abs))
        if img_size is None:
            continue

        w_img, h_img = img_size

        # 添加 image 记录
        img_record = {
            "id": img_id,
            "file_name": image_rel,  # 相对路径
            "width": w_img,
            "height": h_img,
            "_object": obj_name,  # 额外标记所属 object
            "_label": label,      # 额外标记原始标签
        }
        images.append(img_record)

        # 如果是缺陷样本，添加 annotation
        is_anomaly = (label.lower() != "normal")
        if is_anomaly:
            bboxes = []
            n_comp = 0  # mask 内连通域总数（审计用，两种模式都统计）

            if mask_rel:
                mask_abs = data_root / mask_rel
                if not mask_abs.exists():
                    stats["missing_mask"] += 1
                    print(f"    ⚠️ 缺陷样本缺 mask 文件: {mask_rel}（将使用中心区域 fallback）")
                else:
                    bboxes = extract_bboxes(str(mask_abs), img_size, use_cc=use_cc)
                    if not bboxes:
                        stats["empty_mask"] += 1
                        print(f"    ⚠️ mask 为空/无法解析: {mask_rel}（将使用中心区域 fallback）")
                    else:
                        # 连通域计数（无论是否 --cc 都统计，用于审计多缺陷图）
                        try:
                            m_img = np.array(Image.open(mask_abs).convert("L"))
                            _, n_comp = ndimage.label((m_img > 0).astype(np.uint8))
                        except Exception:
                            n_comp = len(bboxes)
            else:
                stats["missing_mask"] += 1
                print(f"    ⚠️ 缺陷样本 CSV 无 mask 列: {image_rel}（将使用中心区域 fallback）")

            if n_comp > 1:
                stats["multi_cc_images"] += 1

            # 如果 mask 全部失败，用图片中心区域作为 fallback（保留历史行为，但显式告警）
            if not bboxes:
                stats["fallback_used"] += 1
                cx, cy = w_img / 2, h_img / 2
                bw, bh = w_img * 0.3, h_img * 0.3
                bboxes = [([cx - bw/2, cy - bh/2, bw, bh], [])]

            for bbox, segmentation in bboxes:
                # 越界检查与裁剪
                x, y, w, h = bbox
                if (x < 0 or y < 0 or x + w > w_img + 1e-6 or y + h > h_img + 1e-6):
                    stats["oob_bbox"] += 1
                    print(f"    ⚠️ bbox 越界 {obj_name}/{image_rel}: {bbox} vs 图像 {img_size} -> 裁剪")
                    x, y = max(0.0, x), max(0.0, y)
                    w = min(w, w_img - x)
                    h = min(h, h_img - y)
                    bbox = [x, y, w, h]

                area = bbox[2] * bbox[3]

                ann_record = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,  # "defect"
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
                annotations.append(ann_record)
                ann_id += 1

        img_id += 1
    
    return images, annotations


def main(args):
    data_root = Path(args.visa_root)
    output_dir = Path(args.output_dir)
    use_cc = args.cc
    check_only = args.check

    print("=" * 60)
    print("VisA CSV → COCO JSON 转换")
    print("=" * 60)
    print(f"  VISA_ROOT = {data_root}")
    print(f"  OUTPUT_DIR = {output_dir}")
    print(f"  TRAIN_RATIO = {args.train_ratio}, SEED = {args.seed}")
    print(f"  连通域分析(--cc): {'开启' if use_cc else '关闭 (与已发表实验 GT 一致)'}")
    print(f"  审计模式(--check): {'开启 (不写文件)' if check_only else '关闭'}")
    
    all_images = []
    all_annotations = []
    
    # 一致性审计计数
    stats = {
        "missing_image": 0,   # CSV 引用的图片文件不存在
        "missing_mask": 0,    # 缺陷样本缺 mask 文件/列
        "empty_mask": 0,      # mask 存在但为空
        "fallback_used": 0,   # 使用中心区域 fallback 的次数
        "oob_bbox": 0,        # bbox 越界并被裁剪的次数
        "multi_cc_images": 0, # mask 含 >1 个连通域的缺陷图数
    }
    
    # 用于生成全局唯一的 ID
    global_img_id = 1
    global_ann_id = 1
    
    for obj_name in OBJECTS:
        print(f"\n📦 处理 object: {obj_name}")
        
        images, annotations = process_object(obj_name, data_root, use_cc=use_cc, stats=stats)
        
        # 重新分配全局 ID
        img_id_map = {}
        for img in images:
            old_id = img["id"]
            img["id"] = global_img_id
            img_id_map[old_id] = global_img_id
            global_img_id += 1
        
        for ann in annotations:
            ann["image_id"] = img_id_map[ann["image_id"]]
            ann["id"] = global_ann_id
            global_ann_id += 1
        
        all_images.extend(images)
        all_annotations.extend(annotations)
        
        n_normal = sum(1 for img in images if img["_label"].lower() == "normal")
        n_anomaly = len(images) - n_normal
        print(f"  ✅ 正常: {n_normal}, 缺陷: {n_anomaly}, 标注数: {len(annotations)}")
    
    # ============ 一致性审计 ============
    n_defect_imgs = sum(1 for img in all_images if img["_label"].lower() != "normal")
    print(f"\n{'='*60}")
    print("🔍 一致性审计:")
    print(f"  缺陷样本图数      : {n_defect_imgs}")
    print(f"  生成标注总数      : {len(all_annotations)}")
    print(f"  缺图片文件        : {stats['missing_image']}")
    print(f"  缺 mask 文件/列   : {stats['missing_mask']}")
    print(f"  空 mask           : {stats['empty_mask']}")
    print(f"  中心区域 fallback : {stats['fallback_used']}")
    print(f"  bbox 越界裁剪     : {stats['oob_bbox']}")
    print(f"  多连通域缺陷图    : {stats['multi_cc_images']} (这些图在 --cc 下会生成额外 bbox)")
    if stats["fallback_used"] or stats["missing_mask"] or stats["empty_mask"]:
        print(f"  ⚠️ 检测到 {stats['fallback_used']} 处 fallback 标注（非真实缺陷区域），"
              f"建议人工核查缺失的 mask 文件。")
    if use_cc:
        expected = n_defect_imgs - stats["fallback_used"] + stats["multi_cc_images"]
        print(f"  --cc 模式理论标注数: {expected} (多连通域图按连通域数拆分)")
    print(f"{'='*60}")
    
    if check_only:
        print("\n✅ 审计完成（--check 模式，未写入任何文件）")
        return
    
    # ============ 划分 train / test ============
    print(f"\n{'='*60}")
    print(f"📊 总计: {len(all_images)} 张图片, {len(all_annotations)} 条标注")
    
    # 按 object 分组，确保 train/test 中都有各 object 的样本
    obj_images = defaultdict(list)
    obj_annotations = defaultdict(list)
    
    img_to_obj = {}
    for img in all_images:
        obj_images[img["_object"]].append(img)
        img_to_obj[img["id"]] = img["_object"]
    
    for ann in all_annotations:
        obj = img_to_obj[ann["image_id"]]
        obj_annotations[obj].append(ann)
    
    train_images = []
    test_images = []
    train_annotations = []
    test_annotations = []
    
    rng = random.Random(args.seed)
    
    for obj_name in OBJECTS:
        imgs = obj_images[obj_name]
        rng.shuffle(imgs)
        split_idx = int(len(imgs) * args.train_ratio)
        
        train_imgs = imgs[:split_idx]
        test_imgs = imgs[split_idx:]
        
        train_img_ids = set(img["id"] for img in train_imgs)
        test_img_ids = set(img["id"] for img in test_imgs)
        
        train_images.extend(train_imgs)
        test_images.extend(test_imgs)
        
        for ann in obj_annotations[obj_name]:
            if ann["image_id"] in train_img_ids:
                train_annotations.append(ann)
            else:
                test_annotations.append(ann)
    
    # ============ 清理额外字段 ============
    for img in train_images + test_images:
        del img["_object"]
        del img["_label"]
    
    # ============ 生成 COCO JSON ============
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": CATEGORIES,
    }
    
    test_json = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": CATEGORIES,
    }
    
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f, indent=2, ensure_ascii=False)
    
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Train: {len(train_images)} 图片, {len(train_annotations)} 标注 → {train_path}")
    print(f"✅ Test:  {len(test_images)} 图片, {len(test_annotations)} 标注 → {test_path}")
    
    # ============ 统计 ============
    print(f"\n{'='*60}")
    print("📋 各 object 统计:")
    for obj_name in OBJECTS:
        n_train = sum(1 for img in train_images if img["file_name"].startswith(obj_name))
        n_test = sum(1 for img in test_images if img["file_name"].startswith(obj_name))
        print(f"  {obj_name:15s}: train={n_train:3d}, test={n_test:3d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main(parse_args())