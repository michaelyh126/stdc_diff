import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image


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
            f'{len(missing)} image patches do not have matching labels. '
            f'First missing: {preview}')

    extra_labels = sorted(set(label_map) - {p.name for p in img_paths})
    if extra_labels:
        preview = ', '.join(extra_labels[:10])
        raise FileNotFoundError(
            f'{len(extra_labels)} label patches do not have matching images. '
            f'First extra: {preview}')

    return pairs


def open_image(path, mode):
    with Image.open(path) as img:
        return img.convert(mode).copy()


def paste_patch(canvas_img, canvas_label, img_path, label_path, x, y,
                tile_size):
    img = open_image(img_path, 'RGB')
    label = open_image(label_path, 'L')
    expected_size = (tile_size, tile_size)

    if img.size != expected_size:
        raise ValueError(f'Image patch is not {expected_size}: {img_path}')
    if label.size != expected_size:
        raise ValueError(f'Label patch is not {expected_size}: {label_path}')

    canvas_img.paste(img, (x, y))
    canvas_label.paste(label, (x, y))


def compose_one(group_pairs, group_idx, out_img_dir, out_label_dir,
                mapping_writer, tile_size, grid_size, save_ext):
    out_size = tile_size * grid_size
    out_name = f'pseudo_{group_idx:04d}{save_ext}'
    canvas_img = Image.new('RGB', (out_size, out_size), color=(0, 0, 0))
    canvas_label = Image.new('L', (out_size, out_size), color=255)

    for tile_idx in range(grid_size * grid_size):
        row = tile_idx // grid_size
        col = tile_idx % grid_size
        x = col * tile_size
        y = row * tile_size

        if tile_idx < len(group_pairs):
            img_path, label_path = group_pairs[tile_idx]
            paste_patch(canvas_img, canvas_label, img_path, label_path, x, y,
                        tile_size)
            image_patch = img_path.name
            label_patch = label_path.name
            is_padding = 0
        else:
            image_patch = 'PAD'
            label_patch = 'PAD'
            is_padding = 1

        mapping_writer.writerow({
            'pseudo_name': out_name,
            'tile_index': tile_idx,
            'row': row,
            'col': col,
            'x': x,
            'y': y,
            'image_patch': image_patch,
            'label_patch': label_patch,
            'is_padding': is_padding,
        })

    canvas_img.save(out_img_dir / out_name)
    canvas_label.save(out_label_dir / out_name)
    return out_name


def compose_dataset(img_dir, label_dir, out_img_dir, out_label_dir,
                    tile_size=500, grid_size=5, seed=0, save_ext='.png'):
    pairs = collect_pairs(img_dir, label_dir)
    patches_per_image = grid_size * grid_size

    if len(pairs) == 0:
        raise RuntimeError(f'No image/label patch pairs found in {img_dir}')

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_img_dir.parent / 'pseudo_mapping.csv'

    num_groups = math.ceil(len(shuffled) / patches_per_image)
    padding_count = num_groups * patches_per_image - len(shuffled)

    with mapping_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'pseudo_name', 'tile_index', 'row', 'col', 'x', 'y',
            'image_patch', 'label_patch', 'is_padding'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for group_idx in range(num_groups):
            start = group_idx * patches_per_image
            end = min(start + patches_per_image, len(shuffled))
            group_pairs = shuffled[start:end]
            out_name = compose_one(group_pairs, group_idx, out_img_dir,
                                   out_label_dir, writer, tile_size,
                                   grid_size, save_ext)
            print(
                f'[DONE] {out_name}: real patches={len(group_pairs)}, '
                f'padding={patches_per_image - len(group_pairs)}')

    print('=' * 60)
    print(f'Total real patches used: {len(shuffled)}')
    print(f'Total padding tiles: {padding_count}')
    print(f'Total pseudo images: {num_groups}')
    print(f'Pseudo image size: {tile_size * grid_size}x{tile_size * grid_size}')
    print(f'Image output dir: {out_img_dir}')
    print(f'Label output dir: {out_label_dir}')
    print(f'Mapping file: {mapping_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Compose aerial_500 test patches into pseudo 2500x2500 '
                    'large images. Real patches are used exactly once; the '
                    'last incomplete image is padded with black/255 tiles.')
    parser.add_argument(
        '--aerial500_root',
        type=str,
        default='/root/autodl-tmp/aerial_500',
        help='Root of aerial_500 dataset.')
    parser.add_argument(
        '--out_root',
        type=str,
        default='/root/autodl-tmp/aerial_500/fake',
        help='Output root for pseudo large test dataset.')
    parser.add_argument('--img_subdir', type=str, default='imgs/test')
    parser.add_argument('--label_subdir', type=str, default='labels/test')
    parser.add_argument('--out_img_subdir', type=str, default='imgs/test')
    parser.add_argument('--out_label_subdir', type=str, default='labels/test')
    parser.add_argument('--tile_size', type=int, default=500)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_ext', type=str, default='.png')
    args = parser.parse_args()

    aerial500_root = Path(args.aerial500_root)
    out_root = Path(args.out_root)

    compose_dataset(
        img_dir=aerial500_root / args.img_subdir,
        label_dir=aerial500_root / args.label_subdir,
        out_img_dir=out_root / args.out_img_subdir,
        out_label_dir=out_root / args.out_label_subdir,
        tile_size=args.tile_size,
        grid_size=args.grid_size,
        seed=args.seed,
        save_ext=args.save_ext)


if __name__ == '__main__':
    main()
