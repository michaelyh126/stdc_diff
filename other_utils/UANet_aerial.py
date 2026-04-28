import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def build_stem_map(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
    return {p.stem: p for p in files}


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)  # H, W, 3
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'RGB 图像维度错误: {arr.shape}')
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 3, H, W


def pil_to_tensor_label(label: Image.Image) -> torch.Tensor:
    arr = np.array(label)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr).contiguous()  # H, W


def tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    arr = x.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(arr)


def tensor_to_pil_label(x: torch.Tensor) -> Image.Image:
    arr = x.contiguous().numpy()
    return Image.fromarray(arr)


def pad_tensor_to_multiple(x: torch.Tensor, tile_size: int, fill_value: int) -> torch.Tensor:
    if x.ndim == 3:
        c, h, w = x.shape
    elif x.ndim == 2:
        h, w = x.shape
    else:
        raise ValueError(f'不支持的维度: {x.shape}')

    new_h = ((h + tile_size - 1) // tile_size) * tile_size
    new_w = ((w + tile_size - 1) // tile_size) * tile_size
    pad_h = new_h - h
    pad_w = new_w - w

    if pad_h == 0 and pad_w == 0:
        return x

    if x.ndim == 3:
        return F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=fill_value)
    else:
        return F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode='constant', value=fill_value).squeeze(0)


def extract_patches_rgb(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    # x: (3, H, W) -> (N, 3, tile, tile)
    c, h, w = x.shape
    patches = x.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, c, tile_size, tile_size)


def extract_patches_label(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    # x: (H, W) -> (N, tile, tile)
    patches = x.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size)
    return patches.contiguous().view(-1, tile_size, tile_size)


def save_one_patch(img_patch_np: np.ndarray,
                   label_patch_np: np.ndarray,
                   out_img_dir: Path,
                   out_label_dir: Path,
                   patch_name: str):
    img_pil = Image.fromarray(img_patch_np)
    label_pil = Image.fromarray(label_patch_np)

    img_pil.save(out_img_dir / patch_name)
    label_pil.save(out_label_dir / patch_name)


def process_one_pair(img_path: Path,
                     label_path: Path,
                     out_img_dir: Path,
                     out_label_dir: Path,
                     tile_size: int = 512,
                     building_threshold: int = 1,
                     save_ext: str = '.png',
                     invalid_fill_value: int = 255,
                     device: str = 'cuda',
                     save_workers: int = 8):
    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path)

    if img.size != label.size:
        raise ValueError(
            f"尺寸不一致: {img_path.name} -> {img.size}, {label_path.name} -> {label.size}"
        )

    img_t = pil_to_tensor_rgb(img)
    label_t = pil_to_tensor_label(label)

    # 先在 CPU 上 pad，再整体搬到 GPU，避免小 tensor 多次搬运
    img_t = pad_tensor_to_multiple(img_t, tile_size, fill_value=invalid_fill_value)
    label_t = pad_tensor_to_multiple(label_t, tile_size, fill_value=invalid_fill_value)

    _, h_pad, w_pad = img_t.shape
    n_h = h_pad // tile_size
    n_w = w_pad // tile_size

    img_t = img_t.to(device, non_blocking=True)
    label_t = label_t.to(device, non_blocking=True)

    img_patches = extract_patches_rgb(img_t, tile_size)       # (N, 3, tile, tile)
    label_patches = extract_patches_label(label_t, tile_size) # (N, tile, tile)

    total = label_patches.shape[0]

    # 255 是无效区域，必须排除
    keep_mask = ((label_patches != invalid_fill_value) &
                 (label_patches >= building_threshold)).flatten(1).any(dim=1)

    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    kept = int(keep_indices.numel())

    if kept == 0:
        return total, kept

    # 只取要保存的 patch，减少 GPU -> CPU 拷贝
    img_keep = img_patches.index_select(0, keep_indices).permute(0, 2, 3, 1).contiguous()
    label_keep = label_patches.index_select(0, keep_indices).contiguous()

    # 一次性搬回 CPU
    img_keep = img_keep.cpu().numpy()      # (K, tile, tile, 3)
    label_keep = label_keep.cpu().numpy()  # (K, tile, tile)
    keep_indices_cpu = keep_indices.cpu().tolist()

    def idx_to_coord(idx: int):
        y_idx = idx // n_w
        x_idx = idx % n_w
        return y_idx * tile_size, x_idx * tile_size

    futures = []
    with ThreadPoolExecutor(max_workers=save_workers) as executor:
        for local_i, global_idx in enumerate(keep_indices_cpu):
            y, x = idx_to_coord(global_idx)
            patch_name = f"{img_path.stem}_{y:04d}_{x:04d}{save_ext}"

            futures.append(
                executor.submit(
                    save_one_patch,
                    img_keep[local_i],
                    label_keep[local_i],
                    out_img_dir,
                    out_label_dir,
                    patch_name
                )
            )

        for f in futures:
            f.result()

    return total, kept


def main():
    parser = argparse.ArgumentParser(
        description='GPU 加速切 patch，padding 无效区域填 255，并行保存 patch'
    )
    parser.add_argument('--img_dir', type=str, required=True, help='原始图像文件夹')
    parser.add_argument('--label_dir', type=str, required=True, help='标签文件夹')
    parser.add_argument('--out_img_dir', type=str, required=True, help='输出图像 patch 文件夹')
    parser.add_argument('--out_label_dir', type=str, required=True, help='输出标签 patch 文件夹')
    parser.add_argument('--tile_size', type=int, default=512, help='patch 大小，默认 512')
    parser.add_argument('--building_threshold', type=int, default=1,
                        help='label 中 >= 该值视为建筑，默认 1')
    parser.add_argument('--invalid_fill_value', type=int, default=255,
                        help='padding 无效区域填充值，默认 255')
    parser.add_argument('--save_ext', type=str, default='.png',
                        help='保存后缀，默认 .png')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda 或 cpu，默认 cuda')
    parser.add_argument('--save_workers', type=int, default=8,
                        help='并行保存线程数，默认 8')
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

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('你指定了 cuda，但当前环境没有可用 GPU')

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
                save_ext=args.save_ext,
                invalid_fill_value=args.invalid_fill_value,
                device=args.device,
                save_workers=args.save_workers
            )
            total_all += total
            kept_all += kept
            print(f'[完成] {stem}: 总 patch = {total}, 保留 = {kept}')
        except Exception as e:
            print(f'[错误] 处理 {stem} 失败: {e}')

    print('=' * 60)
    print('全部完成')
    print(f'总 patch 数: {total_all}')
    print(f'保留 patch 数: {kept_all}')
    print(f'删除空 patch 数: {total_all - kept_all}')
    print(f'图像输出目录: {out_img_dir}')
    print(f'标签输出目录: {out_label_dir}')


if __name__ == '__main__':
    main()
