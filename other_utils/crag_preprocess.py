import argparse
from pathlib import Path
from PIL import Image
import numpy as np

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def convert_mask_to_binary(mask_path: Path, save_path: Path, verbose: bool = False):
    mask = np.array(Image.open(mask_path))

    unique_vals = np.unique(mask)
    binary_mask = (mask != 0).astype(np.uint8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_mask).save(save_path)

    if verbose:
        print(f"[INFO] {mask_path}")
        print(f"       original unique values: {unique_vals.tolist()}")
        print(f"       new unique values: {np.unique(binary_mask).tolist()}")


def process_annotation_dir(src_dir: Path, dst_dir: Path, verbose: bool = False):
    if not src_dir.exists():
        print(f"[WARN] Directory does not exist, skipped: {src_dir}")
        return 0

    files = sorted([p for p in src_dir.iterdir() if p.is_file() and is_image_file(p)])
    if not files:
        print(f"[WARN] No image files found in: {src_dir}")
        return 0

    count = 0
    for p in files:
        save_path = dst_dir / p.name
        convert_mask_to_binary(p, save_path, verbose=verbose)
        count += 1

    print(f"[DONE] Processed {count} files: {src_dir} -> {dst_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert CRAG annotation masks to binary masks (non-zero -> 1).")
    parser.add_argument(
        "--crag_root",
        type=str,
        required=True,
        help="Root directory of CRAG dataset, e.g. /path/to/CRAG"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root directory, e.g. /path/to/CRAG_binary"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print unique values for each annotation"
    )
    args = parser.parse_args()

    crag_root = Path(args.crag_root)
    out_root = Path(args.out_root)

    train_ann_src = crag_root / "train" / "Annotation"
    test_ann_src = crag_root / "valid" / "Annotation"

    train_ann_dst = out_root / "train" / "Annotation"
    test_ann_dst = out_root / "valid" / "Annotation"

    total = 0
    total += process_annotation_dir(train_ann_src, train_ann_dst, verbose=args.verbose)
    total += process_annotation_dir(test_ann_src, test_ann_dst, verbose=args.verbose)

    print(f"\n[ALL DONE] Total processed annotation files: {total}")


if __name__ == "__main__":
    main()
