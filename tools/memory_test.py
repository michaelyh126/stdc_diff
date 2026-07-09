import argparse
import os
import sys
import time

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmseg.models import build_segmentor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Measure CUDA memory for one-image inference')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint file path')
    parser.add_argument('--height', type=int, default=5000, help='input height')
    parser.add_argument('--width', type=int, default=5000, help='input width')
    parser.add_argument('--warmup', type=int, default=1, help='warmup iterations')
    parser.add_argument('--repeat', type=int, default=3, help='measured iterations')
    parser.add_argument(
        '--mode',
        choices=['inference', 'forward-dummy', 'encode-decode'],
        default='inference',
        help='inference path to measure')
    parser.add_argument(
        '--rescale',
        action='store_true',
        help='use rescale=True for inference mode')
    return parser.parse_args()


def build_meta(height, width):
    return dict(
        filename='dummy.png',
        ori_filename='dummy.png',
        img_shape=(height, width, 3),
        ori_shape=(height, width, 3),
        pad_shape=(height, width, 3),
        scale_factor=1.0,
        flip=False,
        flip_direction=None,
        img_norm_cfg=dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False))


def run_once(model, img, meta, mode, rescale):
    if mode == 'inference':
        return model(
            img=[img],
            img_metas=[[meta]],
            return_loss=False,
            rescale=rescale)
    if mode == 'forward-dummy':
        return model.forward_dummy(img)
    if mode == 'encode-decode':
        return model.encode_decode(img, [meta])
    raise ValueError(f'Unsupported mode: {mode}')


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.cuda()
    model.eval()

    img = torch.randn(1, 3, args.height, args.width, device='cuda')
    meta = build_meta(args.height, args.width)

    with torch.no_grad():
        for _ in range(args.warmup):
            output = run_once(model, img, meta, args.mode, args.rescale)
            del output
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        pure_inf_time = 0.0
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = run_once(model, img, meta, args.mode, args.rescale)
            torch.cuda.synchronize()
            pure_inf_time += time.perf_counter() - start
            del output

    allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    current_allocated = torch.cuda.memory_allocated() / 1024 / 1024
    current_reserved = torch.cuda.memory_reserved() / 1024 / 1024
    fps = args.repeat / pure_inf_time if pure_inf_time > 0 else 0.0

    print(f'Mode: {args.mode}')
    print(f'Input: 1x3x{args.height}x{args.width}')
    print(f'Peak CUDA memory allocated: {allocated:.2f} MiB')
    print(f'Peak CUDA memory reserved: {reserved:.2f} MiB')
    print(f'Current CUDA memory allocated: {current_allocated:.2f} MiB')
    print(f'Current CUDA memory reserved: {current_reserved:.2f} MiB')
    print(f'FPS: {fps:.2f}')


if __name__ == '__main__':
    main()
