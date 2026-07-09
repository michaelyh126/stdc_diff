import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


PREFERRED_KEYS = (
    'seg_logits', 'logits', 'pred_logits', 'probs', 'prob', 'scores',
    'score', 'pred', 'preds', 'prediction', 'seg_pred', 'result',
    'results', 'output', 'outputs', 'data')


@dataclass
class Prediction:
    name: str
    pred: np.ndarray
    bg_prob: np.ndarray
    fg_prob: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize three STDC prediction pth files.')
    parser.add_argument('--real-pth', required=True, help='real large image pth')
    parser.add_argument(
        '--pseudo-pth', required=True, help='pseudo large image pth')
    parser.add_argument('--patch-pth', required=True, help='patch pth')
    parser.add_argument(
        '--out-dir', required=True, help='directory to save visualizations')
    parser.add_argument(
        '--real-key', default=None, help='optional key path in real pth')
    parser.add_argument(
        '--pseudo-key', default=None, help='optional key path in pseudo pth')
    parser.add_argument(
        '--patch-key', default=None, help='optional key path in patch pth')
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='sample index when a pth stores a batch or a list')
    parser.add_argument(
        '--layout',
        choices=['auto', 'chw', 'hwc', 'bchw', 'bhwc', 'bhw', 'hw'],
        default='auto',
        help='prediction tensor layout')
    parser.add_argument(
        '--bg-class', type=int, default=0, help='background class index')
    parser.add_argument(
        '--fg-class', type=int, default=1, help='foreground class index')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='threshold for single-channel foreground probabilities')
    parser.add_argument(
        '--real-box',
        nargs=4,
        type=int,
        default=None,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='crop real prediction before comparing with patch')
    parser.add_argument(
        '--pseudo-box',
        nargs=4,
        type=int,
        default=None,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='crop pseudo prediction before comparing with patch')
    parser.add_argument(
        '--cmap', default='magma', help='matplotlib colormap for heatmaps')
    parser.add_argument(
        '--dpi', type=int, default=200, help='saved figure dpi')
    return parser.parse_args()


def get_by_key_path(obj, key_path):
    cur = obj
    for key in key_path.split('.'):
        if isinstance(cur, dict):
            cur = cur[key]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(key)]
        else:
            raise KeyError(f'Cannot access key "{key}" from {type(cur)}')
    return cur


def unwrap_prediction(obj, key_path=None, sample_index=0):
    if key_path is not None:
        return get_by_key_path(obj, key_path)

    if isinstance(obj, dict):
        for key in PREFERRED_KEYS:
            if key in obj:
                return obj[key]
        if 'state_dict' in obj:
            keys = ', '.join(list(obj.keys())[:12])
            raise ValueError(
                'This pth looks like a checkpoint, not a prediction tensor. '
                f'Available keys: {keys}')

        tensor_types = (np.ndarray, list, tuple)
        if torch is not None:
            tensor_types = tensor_types + (torch.Tensor, )
        tensor_keys = [
            key for key, value in obj.items()
            if isinstance(value, tensor_types)
        ]
        if len(tensor_keys) == 1:
            return obj[tensor_keys[0]]
        keys = ', '.join(list(obj.keys())[:20])
        raise ValueError(
            'Cannot infer prediction key automatically. Please pass an '
            f'explicit key. Available keys: {keys}')

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError('Empty prediction list')
        index = min(sample_index, len(obj) - 1)
        return obj[index]

    return obj


def to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
            if arr.dtype != object:
                return arr
        except ValueError:
            pass
        return to_numpy(value[0])
    raise TypeError(f'Unsupported prediction type: {type(value)}')


def apply_layout(arr, layout, sample_index):
    arr = np.asarray(arr)

    if layout == 'bchw':
        return arr[sample_index]
    if layout == 'bhwc':
        return np.moveaxis(arr[sample_index], -1, 0)
    if layout == 'bhw':
        return arr[sample_index]
    if layout == 'chw':
        return arr
    if layout == 'hwc':
        return np.moveaxis(arr, -1, 0)
    if layout == 'hw':
        return arr

    arr = np.squeeze(arr)
    if arr.ndim == 4:
        if arr.shape[1] <= 16:
            return arr[sample_index]
        if arr.shape[-1] <= 16:
            return np.moveaxis(arr[sample_index], -1, 0)
        return arr[sample_index]
    if arr.ndim == 3:
        if arr.shape[0] <= 16:
            return arr
        if arr.shape[-1] <= 16:
            return np.moveaxis(arr, -1, 0)
        return arr[sample_index]
    if arr.ndim == 2:
        return arr
    raise ValueError(f'Unsupported prediction shape: {arr.shape}')


def looks_like_probabilities(arr):
    if arr.size == 0:
        return False
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    return min_value >= -1e-4 and max_value <= 1.0 + 1e-4


def softmax_numpy(arr, axis=0):
    shifted = arr - np.nanmax(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.nansum(exp, axis=axis, keepdims=True) + 1e-12)


def normalize_prediction(name, arr, args):
    arr = apply_layout(arr, args.layout, args.sample_index)

    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        if looks_like_probabilities(arr) and not np.allclose(arr, arr.round()):
            fg_prob = np.clip(arr, 0, 1)
            bg_prob = 1.0 - fg_prob
            pred = (fg_prob >= args.threshold).astype(np.uint8)
        elif looks_like_probabilities(arr) and arr.max() <= 1:
            fg_prob = np.clip(arr, 0, 1)
            bg_prob = 1.0 - fg_prob
            pred = (fg_prob >= args.threshold).astype(np.uint8)
        else:
            pred = arr.astype(np.int64)
            fg_prob = (pred == args.fg_class).astype(np.float32)
            bg_prob = (pred == args.bg_class).astype(np.float32)
        return Prediction(name, pred, bg_prob, fg_prob)

    if arr.ndim != 3:
        raise ValueError(f'{name} must be HxW or CxHxW after layout parsing')

    arr = arr.astype(np.float32)
    channels = arr.shape[0]
    if channels == 1:
        raw = arr[0]
        if looks_like_probabilities(raw):
            fg_prob = np.clip(raw, 0, 1)
        else:
            fg_prob = 1.0 / (1.0 + np.exp(-raw))
        bg_prob = 1.0 - fg_prob
        pred = (fg_prob >= args.threshold).astype(np.uint8)
        return Prediction(name, pred, bg_prob, fg_prob)

    if args.bg_class >= channels or args.fg_class >= channels:
        raise ValueError(
            f'{name} has {channels} channels, but bg_class={args.bg_class} '
            f'and fg_class={args.fg_class}')

    channel_sum = arr.sum(axis=0)
    is_prob = (
        looks_like_probabilities(arr)
        and float(np.nanmean(np.abs(channel_sum - 1.0))) < 0.05)
    if is_prob:
        probs = np.clip(arr, 0, 1)
    else:
        probs = softmax_numpy(arr, axis=0)

    pred = probs.argmax(axis=0).astype(np.uint8)
    bg_prob = probs[args.bg_class].astype(np.float32)
    fg_prob = probs[args.fg_class].astype(np.float32)
    return Prediction(name, pred, bg_prob, fg_prob)


def load_prediction(path, name, key_path, args):
    if torch is None:
        raise ImportError(
            'PyTorch is required to load .pth prediction files. Please run '
            'this script in the same environment you use for STDC/MMCV.')
    obj = torch.load(path, map_location='cpu')
    payload = unwrap_prediction(obj, key_path, args.sample_index)
    arr = to_numpy(payload)
    return normalize_prediction(name, arr, args)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_pred_map(pred, path):
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    rgb[pred == 0] = np.array([0, 0, 0], dtype=np.uint8)
    rgb[pred == 1] = np.array([255, 255, 255], dtype=np.uint8)
    other = (pred != 0) & (pred != 1)
    rgb[other] = np.array([255, 64, 64], dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def save_heatmap(data, path, title, cmap, dpi, vmin=0.0, vmax=1.0):
    height, width = data.shape
    fig_w = max(width / dpi, 3.0)
    fig_h = max(height / dpi, 3.0)
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()


def crop_map(data, box):
    if box is None:
        return data
    x1, y1, x2, y2 = box
    return data[y1:y2, x1:x2]


def resize_map(data, target_shape):
    target_h, target_w = target_shape
    if data.shape == target_shape:
        return data
    image = Image.fromarray(data.astype(np.float32), mode='F')
    image = image.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def residual(left, right):
    if left.shape != right.shape:
        right = resize_map(right, left.shape)
    return np.abs(left.astype(np.float32) - right.astype(np.float32))


def save_prediction_outputs(prediction, out_dir, args):
    save_pred_map(prediction.pred, os.path.join(out_dir,
                                                f'{prediction.name}_pred.png'))
    save_heatmap(
        prediction.fg_prob,
        os.path.join(out_dir, f'{prediction.name}_foreground_heatmap.png'),
        f'{prediction.name} foreground',
        args.cmap,
        args.dpi)
    save_heatmap(
        prediction.bg_prob,
        os.path.join(out_dir, f'{prediction.name}_background_heatmap.png'),
        f'{prediction.name} background',
        args.cmap,
        args.dpi)


def save_summary(predictions, residuals, out_path, args):
    rows = len(predictions) + 1
    cols = 3
    plt.figure(figsize=(cols * 4, rows * 4), dpi=args.dpi)

    for row, prediction in enumerate(predictions):
        plt.subplot(rows, cols, row * cols + 1)
        plt.imshow(prediction.pred, cmap='gray')
        plt.title(f'{prediction.name} pred')
        plt.axis('off')

        plt.subplot(rows, cols, row * cols + 2)
        plt.imshow(prediction.fg_prob, cmap=args.cmap, vmin=0, vmax=1)
        plt.title(f'{prediction.name} foreground')
        plt.axis('off')

        plt.subplot(rows, cols, row * cols + 3)
        plt.imshow(prediction.bg_prob, cmap=args.cmap, vmin=0, vmax=1)
        plt.title(f'{prediction.name} background')
        plt.axis('off')

    for col, (name, value) in enumerate(residuals.items()):
        plt.subplot(rows, cols, len(predictions) * cols + col + 1)
        plt.imshow(value, cmap=args.cmap, vmin=0, vmax=1)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    real = load_prediction(args.real_pth, 'real', args.real_key, args)
    pseudo = load_prediction(args.pseudo_pth, 'pseudo', args.pseudo_key, args)
    patch = load_prediction(args.patch_pth, 'patch', args.patch_key, args)

    for prediction in (real, pseudo, patch):
        save_prediction_outputs(prediction, args.out_dir, args)

    real_for_patch = crop_map(real.fg_prob, args.real_box)
    pseudo_for_patch = crop_map(pseudo.fg_prob, args.pseudo_box)
    residuals = {
        'real_patch_residual':
        residual(real_for_patch, patch.fg_prob),
        'pseudo_patch_residual':
        residual(pseudo_for_patch, patch.fg_prob),
        'real_pseudo_residual':
        residual(real.fg_prob, pseudo.fg_prob),
    }

    for name, value in residuals.items():
        save_heatmap(
            value,
            os.path.join(args.out_dir, f'{name}.png'),
            name,
            args.cmap,
            args.dpi,
            vmin=0.0,
            vmax=1.0)

    save_summary(
        (real, pseudo, patch),
        residuals,
        os.path.join(args.out_dir, 'summary.png'),
        args)

    print(f'Visualization results saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
