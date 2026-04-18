import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def build_stem_map(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
    stem_map = {}
    for p in files:
        stem_map[p.stem] = p
    return stem_map


def pad_to_multiple(img: Image.Image, tile_size: int, fill_value=0) -> Image.Image:
    w, h = img.size
    new_w = ((w + tile_size - 1) // tile_size) * tile_size
    new_h = ((h + tile_size - 1) // tile_size) * tile_size

    if (new_w, new_h) == (w, h):
        return img

    if img.mode in ['RGB', 'RGBA']:
        if isinstance(fill_value, int):
            fill = (fill_value,) * len(img.getbands())
        else:
            fill = fill_value
    else:
        fill = fill_value

    padded = Image.new(img.mode, (new_w, new_h), color=fill)
    padded.paste(img, (0, 0))
    return padded


def has_building(label_patch: Image.Image, threshold: int = 1) -> bool:
    arr = np.array(label_patch)
    return np.any(arr >= threshold)


def save_patch(img_patch: Image.Image, label_patch: Image.Image,
               out_img_dir: Path, out_label_dir: Path,
               base_name: str, y: int, x: int, save_ext: str):
    patch_name = f"{base_name}_{y:04d}_{x:04d}{save_ext}"
    img_patch.save(out_img_dir / patch_name)
    label_patch.save(out_label_dir / patch_name)


def process_one_pair(img_path: Path,
                     label_path: Path,
                     out_img_dir: Path,
                     out_label_dir: Path,
                     tile_size: int = 512,
                     building_threshold: int = 1,
                     save_ext: str = '.png'):
    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path)

    if img.size != label.size:
        raise ValueError(
            f"尺寸不一致: {img_path.name} -> {img.size}, {label_path.name} -> {label.size}"
        )

    # UANet 那种做法：先 pad 到 tile_size 的整数倍
    img = pad_to_multiple(img, tile_size, fill_value=0)
    label = pad_to_multiple(label, tile_size, fill_value=0)

    w, h = img.size
    kept = 0
    total = 0

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            total += 1
            img_patch = img.crop((x, y, x + tile_size, y + tile_size))
            label_patch = label.crop((x, y, x + tile_size, y + tile_size))

            # 没有建筑就不要
            if not has_building(label_patch, threshold=building_threshold):
                continue

            save_patch(
                img_patch, label_patch,
                out_img_dir, out_label_dir,
                img_path.stem, y, x, save_ext
            )
            kept += 1

    return total, kept


def main():
    parser = argparse.ArgumentParser(
        description='将 img/label 文件夹切成 512x512 patch，并删除无建筑 patch'
    )
    parser.add_argument('--img_dir', type=str, required=True, help='原始图像文件夹')
    parser.add_argument('--label_dir', type=str, required=True, help='标签文件夹')
    parser.add_argument('--out_img_dir', type=str, required=True, help='输出图像 patch 文件夹')
    parser.add_argument('--out_label_dir', type=str, required=True, help='输出标签 patch 文件夹')
    parser.add_argument('--tile_size', type=int, default=512, help='patch 大小，默认 512')
    parser.add_argument('--building_threshold', type=int, default=1,
                        help='label 中 >= 该值视为建筑，默认 1')
    parser.add_argument('--save_ext', type=str, default='.png',
                        help='保存后缀，默认 .png')
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_img_dir = Path(args.out_img_dir)
    out_label_dir = Path(args.out_label_dir)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise FileNotFoundError(f'img_dir 不存在: {img_dir}')
    if not label_dir.exists():
        raise FileNotFoundError(f'label_dir 不存在: {label_dir}')

    img_map = build_stem_map(img_dir)
    label_map = build_stem_map(label_dir)

    common_stems = sorted(set(img_map.keys()) & set(label_map.keys()))
    if len(common_stems) == 0:
        raise RuntimeError('img_dir 和 label_dir 中没有同名文件可匹配')

    missing_imgs = sorted(set(label_map.keys()) - set(img_map.keys()))
    missing_labels = sorted(set(img_map.keys()) - set(label_map.keys()))

    if missing_imgs:
        print(f'[警告] 以下 label 找不到对应 image，已跳过: {missing_imgs[:10]}'
              + (' ...' if len(missing_imgs) > 10 else ''))
    if missing_labels:
        print(f'[警告] 以下 image 找不到对应 label，已跳过: {missing_labels[:10]}'
              + (' ...' if len(missing_labels) > 10 else ''))

    total_all = 0
    kept_all = 0

    for stem in common_stems:
        img_path = img_map[stem]
        label_path = label_map[stem]
        try:
            total, kept = process_one_pair(
                img_path=img_path,
                label_path=label_path,
                out_img_dir=out_img_dir,
                out_label_dir=out_label_dir,
                tile_size=args.tile_size,
                building_threshold=args.building_threshold,
                save_ext=args.save_ext
            )
            total_all += total
            kept_all += kept
            print(f'[完成] {stem}: 总 patch = {total}, 保留 = {kept}')
        except Exception as e:
            print(f'[错误] 处理 {stem} 失败: {e}')

    print('=' * 60)
    print(f'全部完成')
    print(f'总 patch 数: {total_all}')
    print(f'保留 patch 数: {kept_all}')
    print(f'删除空 patch 数: {total_all - kept_all}')
    print(f'图像输出目录: {out_img_dir}')
    print(f'标签输出目录: {out_label_dir}')


if __name__ == '__main__':
    main()
