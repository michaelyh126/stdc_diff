import argparse
from pathlib import Path
from PIL import Image
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def collect_images(img_dir: Path):
    return sorted([p for p in img_dir.rglob('*') if p.is_file() and is_image_file(p)])


def compute_mean_std(image_paths):
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for i, img_path in enumerate(image_paths, 1):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.float64)  # HWC, range [0,255]

        h, w, c = img.shape
        assert c == 3, f"Image is not RGB: {img_path}"

        img = img.reshape(-1, 3)  # [H*W, 3]

        pixel_sum += img.sum(axis=0)
        pixel_sq_sum += (img ** 2).sum(axis=0)
        pixel_count += img.shape[0]

        if i % 20 == 0 or i == len(image_paths):
            print(f"[{i}/{len(image_paths)}] processed: {img_path.name}")

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Compute RGB mean/std for CRAG dataset.")
    parser.add_argument(
        "--img_dirs",
        type=str,
        nargs='+',
        required=True,
        help="One or more image directories, e.g. /path/to/CRAG/train/Images /path/to/CRAG/test/Images"
    )
    args = parser.parse_args()

    image_paths = []
    for d in args.img_dirs:
        d = Path(d)
        if not d.exists():
            print(f"[WARN] directory not found: {d}")
            continue
        image_paths.extend(collect_images(d))

    if len(image_paths) == 0:
        print("[ERROR] No images found.")
        return

    print(f"[INFO] Total images: {len(image_paths)}")
    mean, std = compute_mean_std(image_paths)

    print("\n===== Result (0~255 scale) =====")
    print(f"mean={mean.tolist()}")
    print(f"std={std.tolist()}")

    print("\n===== Result (rounded for config) =====")
    print(f"mean={[round(x, 3) for x in mean]}")
    print(f"std={[round(x, 3) for x in std]}")


if __name__ == "__main__":
    main()
