import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_label(path):
    """
    返回:
        label_raw: 原始标签图，裁剪保存时直接用
        fg_mask:   连通域分析用的二值前景图 (0/1)
    """
    label = np.array(Image.open(path))

    if label.ndim == 2:
        fg_mask = (label > 0).astype(np.uint8)
    elif label.ndim == 3:
        fg_mask = (np.any(label > 0, axis=2)).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported label shape: {label.shape}")

    return label, fg_mask


def crop_box_fixed_size(x, y, w, h, img_w, img_h, crop_size=100):
    """
    以目标 bbox 中心为中心，从原图中直接截取固定大小 crop_size x crop_size
    不是 padding；若越界，则整体平移回图像内部
    """
    cx = x + w / 2.0
    cy = y + h / 2.0

    half = crop_size / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # 左上越界，整体往里挪
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0

    # 右下越界，整体往里挪
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 = img_w
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 = img_h

    # 再做一次边界保险
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return x1, y1, x2, y2


def save_array(arr, path):
    Image.fromarray(arr).save(path)


def process_single_pair(
    img_path,
    label_path,
    out_img_dir,
    out_label_dir,
    out_meta_dir,
    min_area=1,
    crop_size=100,
):
    img = load_image(img_path)
    label_raw, fg_mask = load_label(label_path)

    img_h, img_w = img.shape[:2]
    label_h, label_w = label_raw.shape[:2]

    if (img_h != label_h) or (img_w != label_w):
        raise ValueError(
            f"Image and label size mismatch:\n"
            f"img:   {img_path} -> {(img_h, img_w)}\n"
            f"label: {label_path} -> {(label_h, label_w)}"
        )

    num_labels, cc_map, stats, centroids = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )

    stem = Path(img_path).stem
    meta = []
    saved_count = 0

    # 0 是背景
    for comp_id in range(1, num_labels):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[comp_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        # 目标只要宽或高 >= crop_size，就舍弃
        if w >= crop_size or h >= crop_size:
            continue

        x1, y1, x2, y2 = crop_box_fixed_size(
            x, y, w, h, img_w, img_h, crop_size=crop_size
        )

        # 若原图本身过小，无法截出完整 crop_size x crop_size，则跳过
        if (x2 - x1) != crop_size or (y2 - y1) != crop_size:
            continue

        img_crop = img[y1:y2, x1:x2]
        label_crop = label_raw[y1:y2, x1:x2]

        out_name = f"{stem}_obj{comp_id:04d}.png"

        save_array(img_crop, out_img_dir / out_name)
        save_array(label_crop, out_label_dir / out_name)

        cx, cy = centroids[comp_id]

        meta.append(
            {
                "component_id": comp_id,
                "output_name": out_name,
                "orig_bbox_xywh": [x, y, w, h],
                "crop_xyxy": [x1, y1, x2, y2],
                "area": area,
                "centroid_xy": [float(cx), float(cy)],
                "crop_size": [crop_size, crop_size],
            }
        )
        saved_count += 1

    with open(out_meta_dir / f"{stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return saved_count


def build_label_map(label_dir):
    """
    建立 label 文件映射: stem -> full path
    假设 img 和 label 同名
    """
    label_map = {}
    for p in Path(label_dir).iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label_map[p.stem] = p
    return label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="原图文件夹")
    parser.add_argument("--label_dir", type=str, required=True, help="label 文件夹")
    parser.add_argument("--out_img_dir", type=str, required=True, help="裁剪后图像输出文件夹")
    parser.add_argument("--out_label_dir", type=str, required=True, help="裁剪后标签输出文件夹")
    parser.add_argument("--out_meta_dir", type=str, required=True, help="meta 输出文件夹")
    parser.add_argument("--min_area", type=int, default=1, help="忽略面积小于该值的对象")
    parser.add_argument("--crop_size", type=int, default=100, help="固定裁剪尺寸")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_img_dir = Path(args.out_img_dir)
    out_label_dir = Path(args.out_label_dir)
    out_meta_dir = Path(args.out_meta_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"img_dir not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    label_map = build_label_map(label_dir)

    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )

    if len(img_paths) == 0:
        raise RuntimeError(f"No image files found in {img_dir}")

    total_imgs = 0
    total_objects = 0
    missing_labels = []

    for img_path in img_paths:
        stem = img_path.stem
        label_path = label_map.get(stem, None)

        if label_path is None:
            missing_labels.append(img_path.name)
            continue

        try:
            saved_count = process_single_pair(
                img_path=img_path,
                label_path=label_path,
                out_img_dir=out_img_dir,
                out_label_dir=out_label_dir,
                out_meta_dir=out_meta_dir,
                min_area=args.min_area,
                crop_size=args.crop_size,
            )
            total_imgs += 1
            total_objects += saved_count
            print(f"[OK] {img_path.name} -> {saved_count} objects")
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("\n===== Summary =====")
    print(f"Processed images: {total_imgs}")
    print(f"Total objects:    {total_objects}")
    print(f"Output image dir: {out_img_dir}")
    print(f"Output label dir: {out_label_dir}")
    print(f"Output meta dir:  {out_meta_dir}")

    if missing_labels:
        print("\nMissing labels for these images:")
        for name in missing_labels:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
