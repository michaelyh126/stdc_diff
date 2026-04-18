import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """
    读取标签图，并转为二值 mask:
    前景=1，背景=0

    支持:
    - 单通道 0/1
    - 单通道 0/255
    - 多通道图（只要任一通道 > 0 就视为前景）
    """
    img = Image.open(mask_path)
    arr = np.array(img)

    if arr.ndim == 2:
        binary = (arr > 0).astype(np.uint8)
    elif arr.ndim == 3:
        binary = (np.any(arr > 0, axis=2)).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mask shape: {arr.shape} in {mask_path}")

    return binary


def analyze_one_mask(binary_mask: np.ndarray, connectivity: int = 8, min_area: int = 1):
    """
    分析单张 mask

    完整目标定义:
    - 前景连通域
    - 且不接触图像边界

    参数:
    - connectivity: 4 或 8
    - min_area: 小于这个面积的连通域忽略（可用来滤噪点）
    """
    if connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        structure = ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError("connectivity must be 4 or 8")

    labeled, num_components = ndimage.label(binary_mask, structure=structure)
    objects = ndimage.find_objects(labeled)

    # 所有前景像素
    total_positive_pixels = int(binary_mask.sum())

    # 每个连通域面积
    component_sizes = np.bincount(labeled.ravel())
    # component_sizes[0] 是背景，忽略

    all_component_count = 0
    all_component_pixels = 0

    complete_component_count = 0
    complete_component_pixels = 0

    h, w = binary_mask.shape

    for comp_id in range(1, num_components + 1):
        area = int(component_sizes[comp_id])
        if area < min_area:
            continue

        slc = objects[comp_id - 1]
        if slc is None:
            continue

        y_slice, x_slice = slc
        y0, y1 = y_slice.start, y_slice.stop
        x0, x1 = x_slice.start, x_slice.stop

        all_component_count += 1
        all_component_pixels += area

        # 是否接触边界
        touches_border = (y0 == 0) or (x0 == 0) or (y1 == h) or (x1 == w)

        if not touches_border:
            complete_component_count += 1
            complete_component_pixels += area

    return {
        "total_positive_pixels": total_positive_pixels,
        "all_component_count": all_component_count,
        "all_component_pixels": all_component_pixels,
        "complete_component_count": complete_component_count,
        "complete_component_pixels": complete_component_pixels,
    }


def find_mask_files(root_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="统计 train 文件夹中 label 的目标平均像素面积"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default='E:\\aerial\labels\\train',
        help="train 标签文件夹路径"
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        choices=[4, 8],
        help="连通域方式: 4 或 8，默认 8"
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=1,
        help="忽略小于该面积的连通域，默认 1"
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Directory not found: {train_dir}")

    mask_files = find_mask_files(train_dir)
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask files found in: {train_dir}")

    total_positive_pixels = 0

    total_all_component_count = 0
    total_all_component_pixels = 0

    total_complete_component_count = 0
    total_complete_component_pixels = 0

    for mask_path in mask_files:
        try:
            binary_mask = load_binary_mask(mask_path)
            stats = analyze_one_mask(
                binary_mask,
                connectivity=args.connectivity,
                min_area=args.min_area
            )

            total_positive_pixels += stats["total_positive_pixels"]

            total_all_component_count += stats["all_component_count"]
            total_all_component_pixels += stats["all_component_pixels"]

            total_complete_component_count += stats["complete_component_count"]
            total_complete_component_pixels += stats["complete_component_pixels"]

        except Exception as e:
            print(f"[Warning] Skip {mask_path}: {e}")

    avg_pixels_per_all_component = (
        total_all_component_pixels / total_all_component_count
        if total_all_component_count > 0 else 0.0
    )

    avg_pixels_per_complete_component = (
        total_complete_component_pixels / total_complete_component_count
        if total_complete_component_count > 0 else 0.0
    )

    print("=" * 60)
    print(f"Mask folder: {train_dir}")
    print(f"Number of mask files: {len(mask_files)}")
    print(f"Connectivity: {args.connectivity}")
    print(f"Min area: {args.min_area}")
    print("-" * 60)

    print(f"总的像素1数量（全部前景像素）: {total_positive_pixels}")

    print("-" * 60)
    print(f"全部连通域数量: {total_all_component_count}")
    print(f"全部连通域前景像素总数: {total_all_component_pixels}")
    print(f"全部连通域平均像素面积: {avg_pixels_per_all_component:.4f}")

    print("-" * 60)
    print(f"完整目标数量（不接触边界）: {total_complete_component_count}")
    print(f"完整目标前景像素总数: {total_complete_component_pixels}")
    print(f"完整目标平均像素面积: {avg_pixels_per_complete_component:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
