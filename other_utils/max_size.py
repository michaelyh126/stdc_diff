import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_longest_side(img_dir: Path):
    results = []

    for path in sorted(img_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMG_EXTS:
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                longest_side = max(width, height)
                results.append({
                    "filename": path.name,
                    "width": width,
                    "height": height,
                    "longest_side": longest_side,
                })
        except Exception as e:
            print(f"[跳过] {path.name}: {e}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="统计文件夹下图像最长边边长的直方图")
    parser.add_argument("--img_dir", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--bins", type=int, default=30, help="直方图分箱数")
    parser.add_argument("--out_csv", type=str, default="image_longest_side_stats.csv", help="输出CSV路径")
    parser.add_argument("--out_png", type=str, default="image_longest_side_histogram.png", help="输出直方图路径")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"文件夹不存在: {img_dir}")

    df = collect_longest_side(img_dir)
    if df.empty:
        raise RuntimeError("没有找到可用图像")

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.hist(df["longest_side"], bins=args.bins, edgecolor="black")
    plt.xlabel("Longest Side Length")
    plt.ylabel("Number of Images")
    plt.title("Histogram of Image Longest Side Length")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.show()

    print("统计完成")
    print(f"图像数量: {len(df)}")
    print(f"最长边最小值: {df['longest_side'].min()}")
    print(f"最长边最大值: {df['longest_side'].max()}")
    print(f"最长边平均值: {df['longest_side'].mean():.2f}")
    print(f"建议 padding 至少不小于: {df['longest_side'].max()}")
    print(f"CSV已保存到: {args.out_csv}")
    print(f"直方图已保存到: {args.out_png}")


if __name__ == "__main__":
    main()
