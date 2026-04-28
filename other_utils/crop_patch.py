# import argparse
# from pathlib import Path
# from PIL import Image
# import torch
# import numpy as np
#
# IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
#
# def is_image_file(path: Path) -> bool:
#     return path.suffix.lower() in IMG_EXTS
#
# def build_stem_map(folder: Path, suffix_to_strip: str = ''):
#     files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
#     stem_map = {}
#     for p in files:
#         stem = p.stem
#         if suffix_to_strip and stem.endswith(suffix_to_strip):
#             stem = stem[:-len(suffix_to_strip)]
#         stem_map[stem] = p
#     return stem_map
#
# # 默认填充值改成 255
# def pad_to_multiple(img: Image.Image, tile_size: int, fill_value=255) -> Image.Image:
#     w, h = img.size
#     new_w = ((w + tile_size - 1) // tile_size) * tile_size
#     new_h = ((h + tile_size - 1) // tile_size) * tile_size
#     if (new_w, new_h) == (w, h):
#         return img
#     fill = (fill_value,) * len(img.getbands()) if img.mode in ['RGB', 'RGBA'] else fill_value
#     padded = Image.new(img.mode, (new_w, new_h), color=fill)
#     padded.paste(img, (0, 0))
#     return padded
#
# def save_patch(img_patch: Image.Image, label_patch_tensor: torch.Tensor,
#                out_img_dir: Path, out_label_dir: Path,
#                base_name: str, y: int, x: int, save_ext: str):
#     patch_name = f"{base_name}_{y:04d}_{x:04d}{save_ext}"
#     img_patch.save(out_img_dir / patch_name)
#     label_np = label_patch_tensor.cpu().numpy()
#     if label_np.ndim == 2:
#         label_img = Image.fromarray(label_np.astype(np.uint8))
#     else:
#         label_img = Image.fromarray(label_np.astype(np.uint8))
#     label_img.save(out_label_dir / patch_name)
#
# def process_one_pair(img_path: Path, label_path: Path,
#                      out_img_dir: Path, out_label_dir: Path,
#                      tile_size: int = 512, save_ext: str = '.png'):
#     img = Image.open(img_path).convert('RGB')
#     label = Image.open(label_path)
#
#     if img.size != label.size:
#         raise ValueError(f"尺寸不一致: {img_path.name} -> {img.size}, {label_path.name} -> {label.size}")
#
#     # pad 填充为 255
#     img = pad_to_multiple(img, tile_size, fill_value=255)
#     label = pad_to_multiple(label, tile_size, fill_value=255)
#
#     w, h = img.size
#     total, kept = 0, 0
#
#     label_np = np.array(label)
#     label_tensor = torch.from_numpy(label_np).to('cuda')
#
#     for y in range(0, h, tile_size):
#         for x in range(0, w, tile_size):
#             total += 1
#             img_patch = img.crop((x, y, x + tile_size, y + tile_size))
#             if label_tensor.ndim == 2:
#                 label_patch_tensor = label_tensor[y:y+tile_size, x:x+tile_size]
#             else:
#                 label_patch_tensor = label_tensor[y:y+tile_size, x:x+tile_size, :]
#             save_patch(img_patch, label_patch_tensor, out_img_dir, out_label_dir,
#                        img_path.stem, y, x, save_ext)
#             kept += 1
#
#     return total, kept
#
# def main():
#     parser = argparse.ArgumentParser(description='多类别 GPU 优化版切图 512x512 (填充值 255)')
#     parser.add_argument('--img_dir', type=str, required=True)
#     parser.add_argument('--label_dir', type=str, required=True)
#     parser.add_argument('--out_img_dir', type=str, required=True)
#     parser.add_argument('--out_label_dir', type=str, required=True)
#     parser.add_argument('--tile_size', type=int, default=512)
#     parser.add_argument('--save_ext', type=str, default='.png')
#     args = parser.parse_args()
#
#     img_dir = Path(args.img_dir)
#     label_dir = Path(args.label_dir)
#     out_img_dir = Path(args.out_img_dir)
#     out_label_dir = Path(args.out_label_dir)
#
#     out_img_dir.mkdir(parents=True, exist_ok=True)
#     out_label_dir.mkdir(parents=True, exist_ok=True)
#
#     img_map = build_stem_map(img_dir, suffix_to_strip='_sat')
#     label_map = build_stem_map(label_dir, suffix_to_strip='_mask')
#
#     common_stems = sorted(set(img_map.keys()) & set(label_map.keys()))
#     if len(common_stems) == 0:
#         raise RuntimeError('img_dir 和 label_dir 中没有可匹配的文件')
#
#     total_all, kept_all = 0, 0
#     for stem in common_stems:
#         img_path = img_map[stem]
#         label_path = label_map[stem]
#         try:
#             total, kept = process_one_pair(img_path, label_path, out_img_dir, out_label_dir,
#                                            tile_size=args.tile_size,
#                                            save_ext=args.save_ext)
#             total_all += total
#             kept_all += kept
#             print(f'[完成] {stem}: 总 patch = {total}, 保存 = {kept}')
#         except Exception as e:
#             print(f'[错误] {stem} 处理失败: {e}')
#
#     print('='*50)
#     print(f'全部完成')
#     print(f'总 patch 数: {total_all}, 保存 patch 数: {kept_all}')
#     print(f'图像输出目录: {out_img_dir}')
#     print(f'标签输出目录: {out_label_dir}')
#
# if __name__ == '__main__':
#     main()


# import argparse
# from pathlib import Path
# from PIL import Image
# import torch
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
#
# IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
#
#
# def is_image_file(path: Path) -> bool:
#     return path.suffix.lower() in IMG_EXTS
#
#
# def build_stem_map(folder: Path, suffix_to_strip: str = ''):
#     files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
#     stem_map = {}
#     for p in files:
#         stem = p.stem
#         if suffix_to_strip and stem.endswith(suffix_to_strip):
#             stem = stem[:-len(suffix_to_strip)]
#         stem_map[stem] = p
#     return stem_map
#
#
# def pad_array_to_multiple(arr: np.ndarray, tile_size: int, fill_value=255):
#     """
#     arr:
#         image: HWC
#         label: HW 或 HWC
#     """
#     h, w = arr.shape[:2]
#     new_h = ((h + tile_size - 1) // tile_size) * tile_size
#     new_w = ((w + tile_size - 1) // tile_size) * tile_size
#
#     if new_h == h and new_w == w:
#         return arr
#
#     if arr.ndim == 2:
#         padded = np.full((new_h, new_w), fill_value, dtype=arr.dtype)
#         padded[:h, :w] = arr
#     else:
#         c = arr.shape[2]
#         padded = np.full((new_h, new_w, c), fill_value, dtype=arr.dtype)
#         padded[:h, :w, :] = arr
#
#     return padded
#
#
# def save_patch_worker(img_patch_np: np.ndarray,
#                       label_patch_np: np.ndarray,
#                       out_img_dir: Path,
#                       out_label_dir: Path,
#                       patch_name: str):
#     """
#     在线程里保存，避免主线程阻塞在磁盘 IO 上
#     """
#     img_patch = Image.fromarray(img_patch_np.astype(np.uint8))
#     img_patch.save(out_img_dir / patch_name)
#
#     label_patch = Image.fromarray(label_patch_np.astype(np.uint8))
#     label_patch.save(out_label_dir / patch_name)
#
#
# def process_one_pair(img_path: Path,
#                      label_path: Path,
#                      out_img_dir: Path,
#                      out_label_dir: Path,
#                      tile_size: int = 512,
#                      save_ext: str = '.png',
#                      device: str = 'cuda',
#                      save_workers: int = 8,
#                      max_pending: int = 128):
#     """
#     单对图像/标签处理：
#     - CPU 读图
#     - numpy pad
#     - GPU 切 patch
#     - 多线程保存
#     """
#     img = Image.open(img_path).convert('RGB')
#     label = Image.open(label_path)
#
#     if img.size != label.size:
#         raise ValueError(f"尺寸不一致: {img_path.name} -> {img.size}, {label_path.name} -> {label.size}")
#
#     img_np = np.array(img)          # HWC
#     label_np = np.array(label)      # HW 或 HWC
#
#     img_np = pad_array_to_multiple(img_np, tile_size, fill_value=255)
#     label_np = pad_array_to_multiple(label_np, tile_size, fill_value=255)
#
#     h, w = img_np.shape[:2]
#     n_h = h // tile_size
#     n_w = w // tile_size
#
#     # ---- 搬到 GPU ----
#     # 图像: HWC -> CHW
#     img_tensor = torch.from_numpy(img_np).to(device=device, non_blocking=True).permute(2, 0, 1).contiguous()
#
#     if label_np.ndim == 2:
#         label_tensor = torch.from_numpy(label_np).to(device=device, non_blocking=True).contiguous()
#     else:
#         label_tensor = torch.from_numpy(label_np).to(device=device, non_blocking=True).permute(2, 0, 1).contiguous()
#
#     # ---- unfold 切 patch（GPU 上视图操作）----
#     # img_patches: [C, n_h, n_w, tile, tile] -> [n_h, n_w, tile, tile, C]
#     img_patches = img_tensor.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
#     img_patches = img_patches.permute(1, 2, 3, 4, 0).contiguous()
#
#     if label_tensor.ndim == 2:
#         # [n_h, n_w, tile, tile]
#         label_patches = label_tensor.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size).contiguous()
#     else:
#         # [C, n_h, n_w, tile, tile] -> [n_h, n_w, tile, tile, C]
#         label_patches = label_tensor.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
#         label_patches = label_patches.permute(1, 2, 3, 4, 0).contiguous()
#
#     total = n_h * n_w
#     kept = 0
#
#     futures = set()
#
#     with ThreadPoolExecutor(max_workers=save_workers) as executor:
#         for iy in range(n_h):
#             for ix in range(n_w):
#                 y = iy * tile_size
#                 x = ix * tile_size
#                 patch_name = f"{img_path.stem}_{y:04d}_{x:04d}{save_ext}"
#
#                 # 从 GPU 取出当前 patch
#                 img_patch_np = img_patches[iy, ix].cpu().numpy()
#
#                 if label_tensor.ndim == 2:
#                     label_patch_np = label_patches[iy, ix].cpu().numpy()
#                 else:
#                     label_patch_np = label_patches[iy, ix].cpu().numpy()
#
#                 fut = executor.submit(
#                     save_patch_worker,
#                     img_patch_np,
#                     label_patch_np,
#                     out_img_dir,
#                     out_label_dir,
#                     patch_name
#                 )
#                 futures.add(fut)
#                 kept += 1
#
#                 # 控制挂起任务数量，避免内存暴涨
#                 if len(futures) >= max_pending:
#                     done, futures = wait(futures, return_when=FIRST_COMPLETED)
#                     for f in done:
#                         f.result()
#
#         # 等待剩余保存任务结束
#         for f in futures:
#             f.result()
#
#     # 主动释放显存
#     del img_tensor, label_tensor, img_patches, label_patches
#     if device.startswith('cuda'):
#         torch.cuda.empty_cache()
#
#     return total, kept
#
#
# def main():
#     parser = argparse.ArgumentParser(description='GPU 切图 + 多线程保存 patch')
#     parser.add_argument('--img_dir', type=str, required=True)
#     parser.add_argument('--label_dir', type=str, required=True)
#     parser.add_argument('--out_img_dir', type=str, required=True)
#     parser.add_argument('--out_label_dir', type=str, required=True)
#     parser.add_argument('--tile_size', type=int, default=306)
#     parser.add_argument('--save_ext', type=str, default='.png')
#     parser.add_argument('--save_workers', type=int, default=8, help='保存线程数')
#     parser.add_argument('--max_pending', type=int, default=128, help='最多挂起多少个保存任务，防止爆内存')
#     parser.add_argument('--device', type=str, default='cuda')
#     args = parser.parse_args()
#
#     img_dir = Path(args.img_dir)
#     label_dir = Path(args.label_dir)
#     out_img_dir = Path(args.out_img_dir)
#     out_label_dir = Path(args.out_label_dir)
#
#     out_img_dir.mkdir(parents=True, exist_ok=True)
#     out_label_dir.mkdir(parents=True, exist_ok=True)
#
#     img_map = build_stem_map(img_dir, suffix_to_strip='_sat')
#     label_map = build_stem_map(label_dir, suffix_to_strip='_mask')
#
#     common_stems = sorted(set(img_map.keys()) & set(label_map.keys()))
#     if len(common_stems) == 0:
#         raise RuntimeError('img_dir 和 label_dir 中没有可匹配的文件')
#
#     if args.device.startswith('cuda') and not torch.cuda.is_available():
#         raise RuntimeError('你指定了 cuda，但当前环境没有可用 GPU')
#
#     total_all, kept_all = 0, 0
#
#     for stem in common_stems:
#         img_path = img_map[stem]
#         label_path = label_map[stem]
#         try:
#             total, kept = process_one_pair(
#                 img_path=img_path,
#                 label_path=label_path,
#                 out_img_dir=out_img_dir,
#                 out_label_dir=out_label_dir,
#                 tile_size=args.tile_size,
#                 save_ext=args.save_ext,
#                 device=args.device,
#                 save_workers=args.save_workers,
#                 max_pending=args.max_pending
#             )
#             total_all += total
#             kept_all += kept
#             print(f'[完成] {stem}: 总 patch = {total}, 保存 = {kept}')
#         except Exception as e:
#             print(f'[错误] {stem} 处理失败: {e}')
#
#     print('=' * 60)
#     print('全部完成')
#     print(f'总 patch 数: {total_all}, 保存 patch 数: {kept_all}')
#     print(f'图像输出目录: {out_img_dir}')
#     print(f'标签输出目录: {out_label_dir}')
#
#
# if __name__ == '__main__':
#     main()


import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

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


def pad_array_to_multiple(arr: np.ndarray, tile_size: int, fill_value=255):
    """
    arr:
        image: HWC
        label: HW 或 HWC
    """
    h, w = arr.shape[:2]
    new_h = ((h + tile_size - 1) // tile_size) * tile_size
    new_w = ((w + tile_size - 1) // tile_size) * tile_size

    if new_h == h and new_w == w:
        return arr

    if arr.ndim == 2:
        padded = np.full((new_h, new_w), fill_value, dtype=arr.dtype)
        padded[:h, :w] = arr
    else:
        c = arr.shape[2]
        padded = np.full((new_h, new_w, c), fill_value, dtype=arr.dtype)
        padded[:h, :w, :] = arr

    return padded


def should_keep_patch(label_patch_np: np.ndarray, ignore_index: int = 255) -> bool:
    """
    忽略 ignore_index 后：
    - 有效像素为空：舍弃
    - 只有 1 个有效类别：舍弃
    - 有 2 个及以上有效类别：保留
    """
    if label_patch_np.ndim == 3:
        # 正常语义分割标签一般应为 2D
        # 若读出来是 HWC，则默认取第一个通道
        label_patch_np = label_patch_np[..., 0]

    valid = label_patch_np[label_patch_np != ignore_index]
    if valid.size == 0:
        return False

    unique_classes = np.unique(valid)
    return len(unique_classes) >= 2


def save_patch_worker(img_patch_np: np.ndarray,
                      label_patch_np: np.ndarray,
                      out_img_dir: Path,
                      out_label_dir: Path,
                      patch_name: str):
    img_patch = Image.fromarray(img_patch_np.astype(np.uint8))
    img_patch.save(out_img_dir / patch_name)

    label_patch = Image.fromarray(label_patch_np.astype(np.uint8))
    label_patch.save(out_label_dir / patch_name)


def process_one_pair(img_path: Path,
                     label_path: Path,
                     out_img_dir: Path,
                     out_label_dir: Path,
                     tile_size: int = 512,
                     save_ext: str = '.png',
                     device: str = 'cuda',
                     save_workers: int = 8,
                     max_pending: int = 128,
                     drop_single_class: bool = False,
                     ignore_index: int = 255):
    """
    单对图像/标签处理：
    - CPU 读图
    - numpy pad
    - GPU 切 patch
    - 多线程保存
    - 可选：丢弃单类别 patch
    """
    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path)

    if img.size != label.size:
        raise ValueError(f"尺寸不一致: {img_path.name} -> {img.size}, {label_path.name} -> {label.size}")

    img_np = np.array(img)      # HWC
    label_np = np.array(label)  # HW 或 HWC

    img_np = pad_array_to_multiple(img_np, tile_size, fill_value=255)
    label_np = pad_array_to_multiple(label_np, tile_size, fill_value=255)

    h, w = img_np.shape[:2]
    n_h = h // tile_size
    n_w = w // tile_size

    # 搬到 GPU
    img_tensor = torch.from_numpy(img_np).to(device=device, non_blocking=True).permute(2, 0, 1).contiguous()

    if label_np.ndim == 2:
        label_tensor = torch.from_numpy(label_np).to(device=device, non_blocking=True).contiguous()
    else:
        label_tensor = torch.from_numpy(label_np).to(device=device, non_blocking=True).permute(2, 0, 1).contiguous()

    # GPU unfold 切 patch
    # 图像: [C, H, W] -> [n_h, n_w, tile, tile, C]
    img_patches = img_tensor.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
    img_patches = img_patches.permute(1, 2, 3, 4, 0).contiguous()

    if label_tensor.ndim == 2:
        # [H, W] -> [n_h, n_w, tile, tile]
        label_patches = label_tensor.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size).contiguous()
    else:
        # [C, H, W] -> [n_h, n_w, tile, tile, C]
        label_patches = label_tensor.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
        label_patches = label_patches.permute(1, 2, 3, 4, 0).contiguous()

    total = n_h * n_w
    kept = 0
    dropped = 0

    futures = set()

    with ThreadPoolExecutor(max_workers=save_workers) as executor:
        for iy in range(n_h):
            for ix in range(n_w):
                y = iy * tile_size
                x = ix * tile_size
                patch_name = f"{img_path.stem}_{y:04d}_{x:04d}{save_ext}"

                # 从 GPU 取出当前 patch
                img_patch_np = img_patches[iy, ix].cpu().numpy()

                if label_tensor.ndim == 2:
                    label_patch_np = label_patches[iy, ix].cpu().numpy()
                else:
                    label_patch_np = label_patches[iy, ix].cpu().numpy()

                # 可选：过滤单类别 patch
                if drop_single_class:
                    if not should_keep_patch(label_patch_np, ignore_index=ignore_index):
                        dropped += 1
                        continue

                fut = executor.submit(
                    save_patch_worker,
                    img_patch_np,
                    label_patch_np,
                    out_img_dir,
                    out_label_dir,
                    patch_name
                )
                futures.add(fut)
                kept += 1

                # 控制挂起任务数量，避免内存暴涨
                if len(futures) >= max_pending:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()

        for f in futures:
            f.result()

    del img_tensor, label_tensor, img_patches, label_patches
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    return total, kept, dropped


def main():
    parser = argparse.ArgumentParser(description='GPU 切图 + 多线程保存 + 可选单类别 patch 过滤')
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--out_img_dir', type=str, required=True)
    parser.add_argument('--out_label_dir', type=str, required=True)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--save_ext', type=str, default='.png')
    parser.add_argument('--save_workers', type=int, default=8, help='保存线程数')
    parser.add_argument('--max_pending', type=int, default=128, help='最多挂起多少个保存任务，防止爆内存')
    parser.add_argument('--device', type=str, default='cuda')

    # 可开关功能
    parser.add_argument(
        '--drop_single_class',
        action='store_true',
        help='启用后，若 patch 在忽略 ignore_index 后只包含 0 个或 1 个类别，则舍弃'
    )
    parser.add_argument(
        '--ignore_index',
        type=int,
        default=255,
        help='判断类别数时忽略的标签值，默认 255'
    )

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

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('你指定了 cuda，但当前环境没有可用 GPU')

    total_all, kept_all, dropped_all = 0, 0, 0

    for stem in common_stems:
        img_path = img_map[stem]
        label_path = label_map[stem]
        try:
            total, kept, dropped = process_one_pair(
                img_path=img_path,
                label_path=label_path,
                out_img_dir=out_img_dir,
                out_label_dir=out_label_dir,
                tile_size=args.tile_size,
                save_ext=args.save_ext,
                device=args.device,
                save_workers=args.save_workers,
                max_pending=args.max_pending,
                drop_single_class=args.drop_single_class,
                ignore_index=args.ignore_index
            )
            total_all += total
            kept_all += kept
            dropped_all += dropped
            print(f'[完成] {stem}: 总 patch = {total}, 保存 = {kept}, 舍弃 = {dropped}')
        except Exception as e:
            print(f'[错误] {stem} 处理失败: {e}')

    print('=' * 60)
    print('全部完成')
    print(f'总 patch 数: {total_all}')
    print(f'保存 patch 数: {kept_all}')
    print(f'舍弃 patch 数: {dropped_all}')
    print(f'图像输出目录: {out_img_dir}')
    print(f'标签输出目录: {out_label_dir}')


if __name__ == '__main__':
    main()
