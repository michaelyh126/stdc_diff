import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def build_stem_map(folder: Path, suffix_to_strip: str = ''):
    files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
    stem_map = {}
    for p in files:
        stem = p.stem
        if suffix_to_strip and stem.endswith(suffix_to_strip):
            stem = stem[:-len(suffix_to_strip)]
        stem_map[stem] = p
    return stem_map

# 默认填充值改成 255
def pad_to_multiple(img: Image.Image, tile_size: int, fill_value=255) -> Image.Image:
    w, h = img.size
    new_w = ((w + tile_size - 1) // tile_size) * tile_size
    new_h = ((h + tile_size - 1) // tile_size) * tile_size
    if (new_w, new_h) == (w, h):
        return img
    fill = (fill_value,) * len(img.getbands()) if img.mode in ['RGB', 'RGBA'] else fill_value
    padded = Image.new(img.mode, (new_w, new_h), color=fill)
    padded.paste(img, (0, 0))
    return padded

def save_patch(img_patch: Image.Image, label_patch_tensor: torch.Tensor,
               out_img_dir: Path, out_label_dir: Path,
               base_name: str, y: int, x: int, save_ext: str):
    patch_name = f"{base_name}_{y:04d}_{x:04d}{save_ext}"
    img_patch.save(out_img_dir / patch_name)
    label_np = label_patch_tensor.cpu().numpy()
    if label_np.ndim == 2:
        label_img = Image.fromarray(label_np.astype(np.uint8))
    else:
        label_img = Image.fromarray(label_np.astype(np.uint8))
    label_img.save(out_label_dir / patch_name)

def process_one_pair(img_path: Path, label_path: Path,
                     out_img_dir: Path, out_label_dir: Path,
                     tile_size: int = 512, save_ext: str = '.png'):
    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path)

    if img.size != label.size:
        raise ValueError(f"尺寸不一致: {img_path.name} -> {img.size}, {label_path.name} -> {label.size}")

    # pad 填充为 255
    img = pad_to_multiple(img, tile_size, fill_value=255)
    label = pad_to_multiple(label, tile_size, fill_value=255)

    w, h = img.size
    total, kept = 0, 0

    label_np = np.array(label)
    label_tensor = torch.from_numpy(label_np).to('cuda')

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            total += 1
            img_patch = img.crop((x, y, x + tile_size, y + tile_size))
            if label_tensor.ndim == 2:
                label_patch_tensor = label_tensor[y:y+tile_size, x:x+tile_size]
            else:
                label_patch_tensor = label_tensor[y:y+tile_size, x:x+tile_size, :]
            save_patch(img_patch, label_patch_tensor, out_img_dir, out_label_dir,
                       img_path.stem, y, x, save_ext)
            kept += 1

    return total, kept

def main():
    parser = argparse.ArgumentParser(description='多类别 GPU 优化版切图 512x512 (填充值 255)')
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--out_img_dir', type=str, required=True)
    parser.add_argument('--out_label_dir', type=str, required=True)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--save_ext', type=str, default='.png')
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_img_dir = Path(args.out_img_dir)
    out_label_dir = Path(args.out_label_dir)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_map = build_stem_map(img_dir, suffix_to_strip='_sat')
    label_map = build_stem_map(label_dir, suffix_to_strip='_mask')

    common_stems = sorted(set(img_map.keys()) & set(label_map.keys()))
    if len(common_stems) == 0:
        raise RuntimeError('img_dir 和 label_dir 中没有可匹配的文件')

    total_all, kept_all = 0, 0
    for stem in common_stems:
        img_path = img_map[stem]
        label_path = label_map[stem]
        try:
            total, kept = process_one_pair(img_path, label_path, out_img_dir, out_label_dir,
                                           tile_size=args.tile_size,
                                           save_ext=args.save_ext)
            total_all += total
            kept_all += kept
            print(f'[完成] {stem}: 总 patch = {total}, 保存 = {kept}')
        except Exception as e:
            print(f'[错误] {stem} 处理失败: {e}')

    print('='*50)
    print(f'全部完成')
    print(f'总 patch 数: {total_all}, 保存 patch 数: {kept_all}')
    print(f'图像输出目录: {out_img_dir}')
    print(f'标签输出目录: {out_label_dir}')

if __name__ == '__main__':
    main()
