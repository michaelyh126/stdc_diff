import argparse
import csv
from pathlib import Path

from PIL import Image


Image.MAX_IMAGE_PIXELS = None
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def is_image_file(path):
    return path.suffix.lower() in IMG_EXTS


def collect_pairs(img_dir, label_dir):
    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and is_image_file(p)])
    label_map = {
        p.name: p
        for p in label_dir.iterdir()
        if p.is_file() and is_image_file(p)
    }

    pairs = []
    missing = []
    for img_path in img_paths:
        label_path = label_map.get(img_path.name)
        if label_path is None:
            missing.append(img_path.name)
        else:
            pairs.append((img_path, label_path))

    if missing:
        preview = ', '.join(missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} images do not have matching labels. '
            f'First missing: {preview}')

    extra_labels = sorted(set(label_map) - {p.name for p in img_paths})
    if extra_labels:
        preview = ', '.join(extra_labels[:10])
        raise FileNotFoundError(
            f'{len(extra_labels)} labels do not have matching images. '
            f'First extra: {preview}')

    return pairs


def open_image(path, mode):
    with Image.open(path) as img:
        return img.convert(mode).copy()


def split_pair(img_path, label_path, out_img_dir, out_label_dir, mapping_writer,
               tile_size=2500, save_ext='.png'):
    img = open_image(img_path, 'RGB')
    label = open_image(label_path, 'L')

    if img.size != label.size:
        raise ValueError(
            f'Size mismatch: {img_path.name} {img.size}, '
            f'{label_path.name} {label.size}')

    width, height = img.size
    if width % tile_size != 0 or height % tile_size != 0:
        raise ValueError(
            f'Image size must be divisible by {tile_size}: '
            f'{img_path.name} {width}x{height}')

    rows = height // tile_size
    cols = width // tile_size
    saved = 0

    for row in range(rows):
        for col in range(cols):
            x = col * tile_size
            y = row * tile_size
            box = (x, y, x + tile_size, y + tile_size)

            out_name = f'{img_path.stem}_{row:02d}_{col:02d}{save_ext}'
            img.crop(box).save(out_img_dir / out_name)
            label.crop(box).save(out_label_dir / out_name)

            mapping_writer.writerow({
                'source_image': img_path.name,
                'source_label': label_path.name,
                'patch_name': out_name,
                'row': row,
                'col': col,
                'x': x,
                'y': y,
                'tile_size': tile_size,
            })
            saved += 1

    return saved


def process_split(in_root, out_root, split, tile_size=2500, save_ext='.png'):
    img_dir = in_root / 'imgs' / split
    label_dir = in_root / 'labels' / split
    out_img_dir = out_root / 'imgs' / split
    out_label_dir = out_root / 'labels' / split
    mapping_path = out_root / f'{split}_mapping.csv'

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(img_dir, label_dir)
    if not pairs:
        raise RuntimeError(f'No image/label pairs found for split: {split}')

    total_saved = 0
    with mapping_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'source_image', 'source_label', 'patch_name', 'row', 'col', 'x',
            'y', 'tile_size'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (img_path, label_path) in enumerate(pairs, 1):
            saved = split_pair(img_path, label_path, out_img_dir,
                               out_label_dir, writer, tile_size, save_ext)
            total_saved += saved
            print(f'[{split}] {idx}/{len(pairs)} {img_path.name}: {saved}')

    print(f'[DONE] split={split}, source pairs={len(pairs)}, '
          f'patches={total_saved}')
    print(f'[DONE] images: {out_img_dir}')
    print(f'[DONE] labels: {out_label_dir}')
    print(f'[DONE] mapping: {mapping_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Split aerial test 5000x5000 images and masks into '
                    'non-overlapping 2500x2500 patches.')
    parser.add_argument(
        '--aerial_root',
        type=str,
        default='/root/autodl-tmp/aerial',
        help='Root of original aerial dataset.')
    parser.add_argument(
        '--out_root',
        type=str,
        default='/root/autodl-tmp/aerial_2500',
        help='Output root for 2500x2500 patches.')
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['test'],
        help='Dataset splits to process, e.g. test or train test.')
    parser.add_argument('--tile_size', type=int, default=2500)
    parser.add_argument('--save_ext', type=str, default='.png')
    args = parser.parse_args()

    in_root = Path(args.aerial_root)
    out_root = Path(args.out_root)

    for split in args.splits:
        process_split(
            in_root=in_root,
            out_root=out_root,
            split=split,
            tile_size=args.tile_size,
            save_ext=args.save_ext)


if __name__ == '__main__':
    main()
