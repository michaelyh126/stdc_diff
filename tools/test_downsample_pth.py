import argparse
import csv
import os
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
LABEL_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
METRIC_NAMES = (
    'aAcc', 'mIoU', 'mAcc', 'IoU_background', 'IoU_foreground',
    'Acc_background', 'Acc_foreground', 'Dice_foreground', 'TP', 'TN', 'FP',
    'FN')


@dataclass
class Prediction:
    name: str
    pred: np.ndarray
    fg_prob: np.ndarray


@dataclass
class ImageLabelPair:
    img_path: str
    label_path: str
    rel_img_path: str
    rel_label_path: str
    palette: object = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Batch visualize three STDC checkpoints and GT '
                     'differences using the dataset in config.data.test.')
    )
    parser.add_argument(
        '--config',
        required=True,
        help='STDC config file. Used for all checkpoints unless overridden.')
    parser.add_argument(
        '--real-config',
        default=None,
        help='optional config for the real-large checkpoint')
    parser.add_argument(
        '--pseudo-config',
        default=None,
        help='optional config for the pseudo-large checkpoint')
    parser.add_argument(
        '--patch-config',
        default=None,
        help='optional config for the patch checkpoint')

    parser.add_argument(
        '--real-checkpoint',
        required=True,
        help='checkpoint trained/evaluated for real-large image')
    parser.add_argument(
        '--pseudo-checkpoint',
        required=True,
        help='checkpoint trained/evaluated for pseudo-large image')
    parser.add_argument(
        '--patch-checkpoint',
        required=True,
        help='checkpoint trained/evaluated for patch image')

    parser.add_argument(
        '--img-dir',
        default=None,
        help=('directory containing test images. Defaults to '
              'config.data.test.img_dir'))
    parser.add_argument(
        '--label-dir',
        default=None,
        help=('directory containing binary ground-truth labels. Defaults to '
              'config.data.test.ann_dir'))
    parser.add_argument(
        '--out-dir', required=True, help='directory to save visualizations')

    parser.add_argument(
        '--recursive',
        action='store_true',
        help='scan images recursively under img-dir')
    parser.add_argument(
        '--image-exts',
        nargs='+',
        default=IMAGE_EXTS,
        help='image extensions to scan')
    parser.add_argument(
        '--label-exts',
        nargs='+',
        default=LABEL_EXTS,
        help='label extensions to match')
    parser.add_argument(
        '--skip-missing-label',
        action='store_true',
        help='skip images whose labels cannot be found')

    parser.add_argument(
        '--device',
        default='cuda:0',
        help='inference device, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--bg-class', type=int, default=0, help='background class index')
    parser.add_argument(
        '--fg-class',
        type=int,
        default=None,
        help=('foreground class index. Defaults to all classes except '
              '--bg-class'))
    parser.add_argument(
        '--real-box',
        nargs=4,
        type=int,
        default=None,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='crop real prediction before comparing with GT')
    parser.add_argument(
        '--pseudo-box',
        nargs=4,
        type=int,
        default=None,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='crop pseudo prediction before comparing with GT')
    parser.add_argument(
        '--patch-box',
        nargs=4,
        type=int,
        default=None,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='crop patch prediction before comparing with GT')
    parser.add_argument(
        '--cmap',
        default='magma',
        help='matplotlib colormap for foreground heatmaps')
    parser.add_argument(
        '--dpi', type=int, default=200, help='saved figure dpi')
    parser.add_argument(
        '--patch-size',
        nargs='+',
        type=int,
        default=None,
        metavar='SIZE',
        help='draw patch boundary grid, e.g. --patch-size=250 or 250 250')
    parser.add_argument(
        '--patch-line-color',
        default='cyan',
        help='patch boundary line color for matplotlib figures')
    parser.add_argument(
        '--patch-line-width',
        type=float,
        default=0.8,
        help='patch boundary line width')
    parser.add_argument(
        '--over-color',
        nargs=3,
        type=int,
        default=(255, 64, 64),
        metavar=('R', 'G', 'B'),
        help='RGB color for over-predicted foreground pixels')
    parser.add_argument(
        '--under-color',
        nargs=3,
        type=int,
        default=(64, 128, 255),
        metavar=('R', 'G', 'B'),
        help='RGB color for under-predicted foreground pixels')
    parser.add_argument(
        '--true-fg-color',
        nargs=3,
        type=int,
        default=(255, 255, 255),
        metavar=('R', 'G', 'B'),
        help='RGB color for correct foreground pixels in difference maps')
    parser.add_argument(
        '--true-bg-color',
        nargs=3,
        type=int,
        default=(0, 0, 0),
        metavar=('R', 'G', 'B'),
        help='RGB color for correct background pixels in difference maps')
    parser.add_argument(
        '--no-foreground-heatmap',
        action='store_true',
        help='only save prediction masks and GT difference maps')
    parser.add_argument(
        '--no-per-image-visuals',
        action='store_true',
        help='only write metric tables, without per-image png outputs')
    parser.add_argument(
        '--table-digits',
        type=int,
        default=4,
        help='decimal digits used in printed and markdown tables')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_exts(exts):
    normalized = []
    for ext in exts:
        ext = ext.lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized.append(ext)
    return tuple(normalized)


def normalize_patch_size(args):
    if args.patch_size is None:
        return
    if len(args.patch_size) == 1:
        size = args.patch_size[0]
        args.patch_size = (size, size)
        return
    if len(args.patch_size) == 2:
        args.patch_size = tuple(args.patch_size)
        return
    raise ValueError('--patch-size expects one value or two values')


def normalize_rgb(color, name):
    if len(color) != 3:
        raise ValueError(f'{name} expects exactly three RGB values')
    color = tuple(int(v) for v in color)
    if any(v < 0 or v > 255 for v in color):
        raise ValueError(f'{name} RGB values must be in [0, 255]')
    return color


def normalize_args(args):
    args.image_exts = normalize_exts(args.image_exts)
    args.label_exts = normalize_exts(args.label_exts)
    normalize_patch_size(args)
    args.over_color = normalize_rgb(args.over_color, '--over-color')
    args.under_color = normalize_rgb(args.under_color, '--under-color')
    args.true_fg_color = normalize_rgb(args.true_fg_color, '--true-fg-color')
    args.true_bg_color = normalize_rgb(args.true_bg_color, '--true-bg-color')


def import_dataset_runtime():
    import mmcv

    from mmseg.datasets import build_dataset

    return mmcv, build_dataset


def import_mmseg_runtime():
    import mmcv
    import torch
    from mmcv.parallel import collate, scatter

    from mmseg.apis.inference import LoadImage, init_segmentor
    from mmseg.datasets.pipelines import Compose

    return mmcv, torch, collate, scatter, LoadImage, init_segmentor, Compose


def build_config_dataset(config):
    mmcv, build_dataset = import_dataset_runtime()
    cfg = mmcv.Config.fromfile(config)
    if not hasattr(cfg, 'data') or 'test' not in cfg.data:
        raise KeyError(f'Cannot find data.test in config: {config}')
    return build_dataset(cfg.data.test)


def iter_leaf_datasets(dataset):
    if hasattr(dataset, 'datasets'):
        for child in dataset.datasets:
            yield from iter_leaf_datasets(child)
        return
    if hasattr(dataset, 'dataset'):
        yield from iter_leaf_datasets(dataset.dataset)
        return
    yield dataset


def join_data_path(prefix, path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path) or prefix is None:
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(prefix, path))


def safe_relpath(path, start):
    if start is None:
        return os.path.basename(path)
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.basename(path)


def replace_suffix(path, old_suffix, new_suffix):
    if old_suffix is None or new_suffix is None or old_suffix == '':
        return None
    if path.lower().endswith(old_suffix.lower()):
        return path[:-len(old_suffix)] + new_suffix
    return None


def init_model(config, checkpoint, device):
    _, _, _, _, _, init_segmentor, _ = import_mmseg_runtime()
    return init_segmentor(config, checkpoint, device=device)


def prepare_data(model, img_path):
    _, torch, collate, scatter, LoadImage, _, Compose = import_mmseg_runtime()
    cfg = model.cfg
    device = next(model.parameters()).device
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    data = dict(img=img_path)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    assert torch is not None
    return data


def infer_prob(model, img_path):
    _, torch, _, _, _, _, _ = import_mmseg_runtime()
    data = prepare_data(model, img_path)
    imgs = data['img']
    img_metas = data['img_metas']

    with torch.no_grad():
        prob = model.inference(imgs[0], img_metas[0], rescale=True)
        for aug_idx in range(1, len(imgs)):
            prob += model.inference(
                imgs[aug_idx], img_metas[aug_idx], rescale=True)
        prob = prob / len(imgs)

    return prob[0].detach().cpu().float().numpy()


def read_image_rgb(path):
    mmcv, _, _, _, _, _, _ = import_mmseg_runtime()
    img = mmcv.imread(path, flag='color')
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return mmcv.bgr2rgb(img)


def label_to_binary(label, args):
    return (label != args.bg_class).astype(np.uint8)


def rgb_label_to_class(rgb, palette):
    if palette is None:
        return None

    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        return None

    label = np.zeros(rgb.shape[:2], dtype=np.uint8)
    matched = np.zeros(rgb.shape[:2], dtype=bool)
    for class_id, color in enumerate(palette):
        mask = np.all(rgb == color, axis=-1)
        label[mask] = class_id
        matched |= mask

    if not matched.all():
        label[~matched] = np.any(rgb[~matched] != 0, axis=-1).astype(
            np.uint8)
    return label


def read_groundtruth(path, args, palette=None):
    gt = np.asarray(Image.open(path))
    if gt.ndim == 3:
        rgb = gt[:, :, :3]
        if np.all(rgb[:, :, 0] == rgb[:, :, 1]) and np.all(
                rgb[:, :, 0] == rgb[:, :, 2]):
            gt = rgb[:, :, 0]
        else:
            mapped = rgb_label_to_class(rgb, palette)
            if mapped is not None:
                gt = mapped
            else:
                return np.any(rgb != 0, axis=-1).astype(np.uint8)
    return label_to_binary(gt, args)


def model_to_prediction(model, name, img_path, args):
    prob = infer_prob(model, img_path)
    if args.bg_class >= prob.shape[0]:
        raise ValueError(
            f'{name} output has {prob.shape[0]} classes, but bg_class='
            f'{args.bg_class}')
    if args.fg_class is not None and args.fg_class >= prob.shape[0]:
        raise ValueError(
            f'{name} output has {prob.shape[0]} classes, but fg_class='
            f'{args.fg_class}')

    pred = prob.argmax(axis=0).astype(np.uint8)
    if args.fg_class is None:
        fg_indices = [i for i in range(prob.shape[0]) if i != args.bg_class]
        fg_prob = prob[fg_indices].sum(axis=0).astype(np.float32)
    else:
        fg_prob = prob[args.fg_class].astype(np.float32)
    return Prediction(name, pred, fg_prob)


def find_image_files(img_dir, image_exts, recursive):
    image_paths = []
    if recursive:
        for root, _, files in os.walk(img_dir):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in image_exts:
                    image_paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(img_dir):
            path = os.path.join(img_dir, filename)
            if os.path.isfile(path):
                if os.path.splitext(filename)[1].lower() in image_exts:
                    image_paths.append(path)
    return sorted(image_paths)


def unique_existing_path(candidates):
    seen = set()
    for path in candidates:
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized
    return None


def find_label_path(img_path,
                    img_dir,
                    label_dir,
                    label_exts,
                    img_suffix=None,
                    seg_map_suffix=None):
    rel_path = safe_relpath(img_path, img_dir)
    rel_root, rel_ext = os.path.splitext(rel_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]

    candidates = []
    suffix_rel_path = replace_suffix(rel_path, img_suffix, seg_map_suffix)
    if suffix_rel_path is not None:
        candidates.append(os.path.join(label_dir, suffix_rel_path))

    suffix_stem = replace_suffix(os.path.basename(img_path), img_suffix,
                                 seg_map_suffix)
    if suffix_stem is not None:
        candidates.append(os.path.join(label_dir, suffix_stem))

    if stem.endswith('_sat'):
        mask_stem = stem[:-len('_sat')] + '_mask'
        for ext in label_exts:
            candidates.append(os.path.join(label_dir, mask_stem + ext))

    candidates.append(os.path.join(label_dir, rel_path))
    if rel_ext.lower() in label_exts:
        candidates.append(os.path.join(label_dir, rel_path))
    for ext in label_exts:
        candidates.append(os.path.join(label_dir, rel_root + ext))
    for ext in label_exts:
        candidates.append(os.path.join(label_dir, stem + ext))

    return unique_existing_path(candidates)


def image_label_pairs_from_dataset(dataset, args):
    pairs = []
    for leaf_dataset in iter_leaf_datasets(dataset):
        config_img_dir = getattr(leaf_dataset, 'img_dir', None)
        config_label_dir = getattr(leaf_dataset, 'ann_dir', None)
        img_dir = args.img_dir or config_img_dir
        label_dir = args.label_dir or config_label_dir
        img_suffix = getattr(leaf_dataset, 'img_suffix', None)
        seg_map_suffix = getattr(leaf_dataset, 'seg_map_suffix', None)
        img_infos = getattr(leaf_dataset, 'img_infos', None)
        palette = getattr(leaf_dataset, 'PALETTE', None)

        if img_dir is None:
            raise ValueError('Cannot resolve image directory from config. '
                             'Please provide --img-dir.')
        if label_dir is None:
            raise ValueError('Cannot resolve label directory from config. '
                             'Please provide --label-dir.')

        if img_infos is None:
            image_paths = find_image_files(img_dir, args.image_exts,
                                           args.recursive)
            for img_path in image_paths:
                label_path = find_label_path(img_path, img_dir, label_dir,
                                             args.label_exts, img_suffix,
                                             seg_map_suffix)
                rel_img_path = safe_relpath(img_path, img_dir)
                rel_label_path = (safe_relpath(label_path, label_dir)
                                  if label_path is not None else '')
                pairs.append(
                    ImageLabelPair(img_path, label_path, rel_img_path,
                                   rel_label_path, palette))
            continue

        for img_info in img_infos:
            filename = img_info['filename']
            img_path = join_data_path(img_dir, filename)
            ann = img_info.get('ann', {})
            seg_map = ann.get('seg_map') if ann is not None else None
            label_path = join_data_path(label_dir, seg_map)

            if label_path is None or not os.path.exists(label_path):
                label_path = find_label_path(img_path, img_dir, label_dir,
                                             args.label_exts, img_suffix,
                                             seg_map_suffix)

            rel_img_path = safe_relpath(img_path, img_dir)
            rel_label_path = (safe_relpath(label_path, label_dir)
                              if label_path is not None else '')
            pairs.append(
                ImageLabelPair(img_path, label_path, rel_img_path,
                               rel_label_path, palette))

    return pairs


def safe_output_name(img_path, img_dir):
    return safe_output_name_from_rel(safe_relpath(img_path, img_dir))


def safe_output_name_from_rel(rel_path):
    rel_root = os.path.splitext(rel_path)[0]
    rel_root = rel_root.replace(os.sep, '_').replace('/', '_').replace('\\',
                                                                        '_')
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', rel_root)


def save_rgb_image(image_rgb, path):
    Image.fromarray(image_rgb.astype(np.uint8)).save(path)


def save_pred_map(pred, path):
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    rgb[pred == 0] = np.array([0, 0, 0], dtype=np.uint8)
    rgb[pred == 1] = np.array([255, 255, 255], dtype=np.uint8)
    other = (pred != 0) & (pred != 1)
    rgb[other] = np.array([255, 64, 64], dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def save_binary_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode='L').save(path)


def draw_patch_grid(ax, shape, args):
    if args.patch_size is None:
        return
    patch_h, patch_w = args.patch_size
    if patch_h <= 0 or patch_w <= 0:
        return
    height, width = shape[:2]
    for x in range(patch_w, width, patch_w):
        ax.axvline(
            x - 0.5,
            color=args.patch_line_color,
            linewidth=args.patch_line_width)
    for y in range(patch_h, height, patch_h):
        ax.axhline(
            y - 0.5,
            color=args.patch_line_color,
            linewidth=args.patch_line_width)


def save_image_with_grid(image, path, args, title=None):
    height, width = image.shape[:2]
    plt.figure(figsize=(max(width / args.dpi, 3.0),
                        max(height / args.dpi, 3.0)),
               dpi=args.dpi)
    ax = plt.gca()
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    draw_patch_grid(ax, image.shape, args)
    plt.tight_layout()
    plt.savefig(path, dpi=args.dpi, bbox_inches='tight')
    plt.close()


def save_heatmap(data, path, title, cmap, dpi, args=None, vmin=0.0, vmax=1.0):
    height, width = data.shape
    fig_w = max(width / dpi, 3.0)
    fig_h = max(height / dpi, 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    if args is not None:
        draw_patch_grid(ax, data.shape, args)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_mask_with_grid(mask, path, args, title=None):
    plt.figure(figsize=(max(mask.shape[1] / args.dpi, 3.0),
                        max(mask.shape[0] / args.dpi, 3.0)),
               dpi=args.dpi)
    ax = plt.gca()
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    draw_patch_grid(ax, mask.shape, args)
    plt.tight_layout()
    plt.savefig(path, dpi=args.dpi, bbox_inches='tight')
    plt.close()


def crop_map(data, box):
    if box is None:
        return data
    x1, y1, x2, y2 = box
    return data[y1:y2, x1:x2]


def resize_rgb(image, target_shape):
    target_h, target_w = target_shape
    if image.shape[:2] == target_shape:
        return image
    pil_img = Image.fromarray(image.astype(np.uint8))
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(pil_img)


def resize_label(data, target_shape):
    target_h, target_w = target_shape
    if data.shape == target_shape:
        return data
    image = Image.fromarray(data.astype(np.uint8), mode='L')
    image = image.resize((target_w, target_h), Image.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def align_prediction_to_gt(pred, gt_shape, box=None):
    pred = crop_map(pred, box)
    return resize_label(pred, gt_shape)


def prediction_box(args, name):
    if name == 'real':
        return args.real_box
    if name == 'pseudo':
        return args.pseudo_box
    if name == 'patch':
        return args.patch_box
    return None


def prediction_to_binary(pred, args):
    if args.fg_class is None:
        return (pred != args.bg_class).astype(np.uint8)
    return (pred == args.fg_class).astype(np.uint8)


def compute_binary_metrics(pred, gt, args):
    pred = prediction_to_binary(pred, args)
    gt = (gt > 0).astype(np.uint8)

    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    total = tp + tn + fp + fn

    eps = 1e-12
    bg_iou = tn / max(tn + fp + fn, eps)
    fg_iou = tp / max(tp + fp + fn, eps)
    bg_acc = tn / max(tn + fp, eps)
    fg_acc = tp / max(tp + fn, eps)
    aacc = (tp + tn) / max(total, eps)
    dice_fg = (2 * tp) / max(2 * tp + fp + fn, eps)

    return dict(
        aAcc=aacc,
        mIoU=(bg_iou + fg_iou) * 0.5,
        mAcc=(bg_acc + fg_acc) * 0.5,
        IoU_background=bg_iou,
        IoU_foreground=fg_iou,
        Acc_background=bg_acc,
        Acc_foreground=fg_acc,
        Dice_foreground=dice_fg,
        TP=tp,
        TN=tn,
        FP=fp,
        FN=fn)


def build_gt_diff_map(pred, gt, args):
    pred = prediction_to_binary(pred, args)
    gt = (gt > 0).astype(np.uint8)

    over = (pred == 1) & (gt == 0)
    under = (pred == 0) & (gt == 1)
    true_fg = (pred == 1) & (gt == 1)
    true_bg = (pred == 0) & (gt == 0)

    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[true_bg] = np.array(args.true_bg_color, dtype=np.uint8)
    rgb[true_fg] = np.array(args.true_fg_color, dtype=np.uint8)
    rgb[over] = np.array(args.over_color, dtype=np.uint8)
    rgb[under] = np.array(args.under_color, dtype=np.uint8)
    return rgb


def save_prediction_outputs(prediction, out_dir, args):
    pred_path = os.path.join(out_dir, f'{prediction.name}_pred.png')
    save_pred_map(prediction.pred, pred_path)
    if args.patch_size is not None:
        grid_path = os.path.join(
            out_dir, f'{prediction.name}_pred_patch_grid.png')
        save_mask_with_grid(prediction.pred, grid_path, args)
    if not args.no_foreground_heatmap:
        save_heatmap(
            prediction.fg_prob,
            os.path.join(out_dir,
                         f'{prediction.name}_foreground_heatmap.png'),
            f'{prediction.name} foreground',
            args.cmap,
            args.dpi,
            args=args)


def save_legend(ax, args):
    def to_float_rgb(color):
        return tuple(v / 255.0 for v in color)

    patches = [
        mpatches.Patch(
            color=to_float_rgb(args.over_color), label='over prediction'),
        mpatches.Patch(
            color=to_float_rgb(args.under_color), label='under prediction'),
        mpatches.Patch(
            color=to_float_rgb(args.true_fg_color), label='correct foreground'),
        mpatches.Patch(
            color=to_float_rgb(args.true_bg_color), label='correct background'),
    ]
    ax.legend(handles=patches, loc='center', frameon=False)
    ax.set_title('difference colors')
    ax.axis('off')


def save_summary(image_rgb, gt, aligned_predictions, diff_maps, metrics,
                 out_path, args):
    rows = len(aligned_predictions) + 1
    cols = 3
    plt.figure(figsize=(cols * 4, rows * 4), dpi=args.dpi)

    image_for_gt = resize_rgb(image_rgb, gt.shape)

    plt.subplot(rows, cols, 1)
    ax = plt.gca()
    ax.imshow(image_for_gt)
    ax.set_title('image')
    ax.axis('off')
    draw_patch_grid(ax, image_for_gt.shape, args)

    plt.subplot(rows, cols, 2)
    ax = plt.gca()
    ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
    ax.set_title('ground truth')
    ax.axis('off')
    draw_patch_grid(ax, gt.shape, args)

    plt.subplot(rows, cols, 3)
    save_legend(plt.gca(), args)

    for row, (name, pred) in enumerate(aligned_predictions.items()):
        offset = (row + 1) * cols
        metric = metrics[name]
        title_suffix = f" aAcc={metric['aAcc']:.3f}, mIoU={metric['mIoU']:.3f}"

        plt.subplot(rows, cols, offset + 1)
        ax = plt.gca()
        ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{name} pred{title_suffix}')
        ax.axis('off')
        draw_patch_grid(ax, pred.shape, args)

        plt.subplot(rows, cols, offset + 2)
        ax = plt.gca()
        if args.no_foreground_heatmap:
            ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
            ax.set_title('ground truth')
        else:
            fg_like = prediction_to_binary(pred, args).astype(np.float32)
            ax.imshow(fg_like, cmap=args.cmap, vmin=0, vmax=1)
            ax.set_title(f'{name} foreground mask')
        ax.axis('off')
        draw_patch_grid(ax, pred.shape, args)

        plt.subplot(rows, cols, offset + 3)
        ax = plt.gca()
        ax.imshow(diff_maps[name])
        ax.set_title(f'{name} vs GT')
        ax.axis('off')
        draw_patch_grid(ax, diff_maps[name].shape, args)

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()


def process_one_image(img_path,
                      label_path,
                      image_out_dir,
                      models,
                      args,
                      palette=None):
    image_rgb = read_image_rgb(img_path)
    gt = read_groundtruth(label_path, args, palette)

    predictions = [
        model_to_prediction(model, name, img_path, args)
        for name, model in models.items()
    ]

    aligned_predictions = {}
    diff_maps = {}
    metrics = {}

    if not args.no_per_image_visuals:
        ensure_dir(image_out_dir)
        save_rgb_image(image_rgb, os.path.join(image_out_dir, 'image.png'))
        if args.patch_size is not None:
            save_image_with_grid(
                image_rgb,
                os.path.join(image_out_dir, 'image_patch_grid.png'),
                args,
                title='image')
        save_binary_mask(gt, os.path.join(image_out_dir, 'groundtruth.png'))
        if args.patch_size is not None:
            save_mask_with_grid(
                gt,
                os.path.join(image_out_dir, 'groundtruth_patch_grid.png'),
                args,
                title='ground truth')

    for prediction in predictions:
        if not args.no_per_image_visuals:
            save_prediction_outputs(prediction, image_out_dir, args)

        box = prediction_box(args, prediction.name)
        pred_for_gt = align_prediction_to_gt(prediction.pred, gt.shape, box)
        aligned_predictions[prediction.name] = pred_for_gt

        metrics[prediction.name] = compute_binary_metrics(
            pred_for_gt, gt, args)

        if not args.no_per_image_visuals:
            if box is not None or pred_for_gt.shape != prediction.pred.shape:
                save_pred_map(
                    pred_for_gt,
                    os.path.join(image_out_dir,
                                 f'{prediction.name}_pred_for_gt.png'))

            diff_map = build_gt_diff_map(pred_for_gt, gt, args)
            diff_maps[prediction.name] = diff_map
            save_rgb_image(
                diff_map,
                os.path.join(image_out_dir,
                             f'{prediction.name}_gt_diff.png'))
            if args.patch_size is not None:
                save_image_with_grid(
                    diff_map,
                    os.path.join(
                        image_out_dir,
                        f'{prediction.name}_gt_diff_patch_grid.png'),
                    args,
                    title=f'{prediction.name} vs GT')

    if not args.no_per_image_visuals:
        save_summary(image_rgb, gt, aligned_predictions, diff_maps, metrics,
                     os.path.join(image_out_dir, 'summary.png'), args)

    return metrics


def metric_to_string(value, digits):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f'{float(value):.{digits}f}'


def build_wide_row(rel_img_path, rel_label_path, image_out_dir, metrics):
    row = {
        'image': rel_img_path,
        'label': rel_label_path,
        'visual_dir': image_out_dir,
    }
    for model_name, model_metrics in metrics.items():
        for metric_name in METRIC_NAMES:
            row[f'{model_name}_{metric_name}'] = model_metrics[metric_name]
    return row


def build_long_rows(rel_img_path, rel_label_path, image_out_dir, metrics):
    rows = []
    for model_name, model_metrics in metrics.items():
        row = {
            'image': rel_img_path,
            'label': rel_label_path,
            'model': model_name,
            'visual_dir': image_out_dir,
        }
        for metric_name in METRIC_NAMES:
            row[metric_name] = model_metrics[metric_name]
        rows.append(row)
    return rows


def table_fieldnames(model_names):
    fields = ['image', 'label', 'visual_dir']
    for model_name in model_names:
        for metric_name in METRIC_NAMES:
            fields.append(f'{model_name}_{metric_name}')
    return fields


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(path, rows, headers, digits):
    lines = markdown_table_lines(rows, headers, digits)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def markdown_table_lines(rows, headers, digits):
    def format_cell(value):
        if isinstance(value, float):
            return f'{value:.{digits}f}'
        if isinstance(value, np.floating):
            return f'{float(value):.{digits}f}'
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    formatted_rows = [[format_cell(row.get(header, '')) for header in headers]
                      for row in rows]
    widths = [len(header) for header in headers]
    for row in formatted_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    header_line = '| ' + ' | '.join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)) + ' |'
    sep_line = '| ' + ' | '.join('-' * widths[idx]
                                  for idx in range(len(headers))) + ' |'
    body_lines = [
        '| ' + ' | '.join(value.ljust(widths[idx])
                          for idx, value in enumerate(row)) + ' |'
        for row in formatted_rows
    ]
    return [header_line, sep_line] + body_lines


def compact_table_rows(wide_rows, model_names):
    compact_rows = []
    for row in wide_rows:
        compact = {'image': row['image']}
        for model_name in model_names:
            for metric_name in ('aAcc', 'mIoU', 'IoU_foreground',
                                'Dice_foreground', 'FP', 'FN'):
                compact[f'{model_name}_{metric_name}'] = row[
                    f'{model_name}_{metric_name}']
        compact_rows.append(compact)
    return compact_rows


def write_summary(path, long_rows, model_names):
    summary_rows = []
    for model_name in model_names:
        model_rows = [row for row in long_rows if row['model'] == model_name]
        if len(model_rows) == 0:
            continue

        summary = {'model': model_name, 'num_images': len(model_rows)}
        for metric_name in METRIC_NAMES:
            values = [row[metric_name] for row in model_rows]
            if metric_name in ('TP', 'TN', 'FP', 'FN'):
                summary[metric_name] = int(np.sum(values))
            else:
                summary[metric_name] = float(np.mean(values))
        summary_rows.append(summary)

    write_csv(path, summary_rows, ['model', 'num_images'] + list(METRIC_NAMES))


def main():
    args = parse_args()
    normalize_args(args)
    ensure_dir(args.out_dir)
    dataset = build_config_dataset(args.config)
    image_label_pairs = image_label_pairs_from_dataset(dataset, args)

    real_config = args.real_config or args.config
    pseudo_config = args.pseudo_config or args.config
    patch_config = args.patch_config or args.config

    models = {
        'real':
        init_model(real_config, args.real_checkpoint, args.device),
        'pseudo':
        init_model(pseudo_config, args.pseudo_checkpoint, args.device),
        'patch':
        init_model(patch_config, args.patch_checkpoint, args.device),
    }

    if len(image_label_pairs) == 0:
        raise FileNotFoundError(
            f'No images found from config data.test: {args.config}')

    wide_rows = []
    long_rows = []
    missing_labels = []
    per_image_root = os.path.join(args.out_dir, 'per_image')
    if not args.no_per_image_visuals:
        ensure_dir(per_image_root)

    for index, pair in enumerate(image_label_pairs, start=1):
        img_path = pair.img_path
        label_path = pair.label_path
        rel_img_path = pair.rel_img_path
        if label_path is None:
            missing_labels.append(rel_img_path)
            if args.skip_missing_label:
                print(f'[{index}/{len(image_label_pairs)}] '
                      f'skip missing label: '
                      f'{rel_img_path}')
                continue
            raise FileNotFoundError(
                f'Cannot find label for image: {img_path}')

        rel_label_path = pair.rel_label_path
        output_name = safe_output_name_from_rel(rel_img_path)
        image_out_dir = os.path.join(per_image_root, output_name)
        print(f'[{index}/{len(image_label_pairs)}] processing {rel_img_path}')

        metrics = process_one_image(img_path, label_path, image_out_dir, models,
                                    args, pair.palette)
        wide_rows.append(
            build_wide_row(rel_img_path, rel_label_path, image_out_dir,
                           metrics))
        long_rows.extend(
            build_long_rows(rel_img_path, rel_label_path, image_out_dir,
                            metrics))

    if len(wide_rows) == 0:
        raise RuntimeError('No image was processed successfully.')

    model_names = list(models.keys())
    wide_fields = table_fieldnames(model_names)
    long_fields = ['image', 'label', 'model', 'visual_dir'] + list(
        METRIC_NAMES)

    write_csv(os.path.join(args.out_dir, 'metrics_table.csv'), wide_rows,
              wide_fields)
    write_csv(os.path.join(args.out_dir, 'metrics_long.csv'), long_rows,
              long_fields)
    write_summary(os.path.join(args.out_dir, 'metrics_summary.csv'), long_rows,
                  model_names)

    compact_rows = compact_table_rows(wide_rows, model_names)
    compact_headers = list(compact_rows[0].keys())
    write_markdown_table(os.path.join(args.out_dir, 'metrics_table.md'),
                         compact_rows, compact_headers, args.table_digits)

    print('\nMetrics table:')
    for line in markdown_table_lines(compact_rows, compact_headers,
                                     args.table_digits):
        print(line)

    if missing_labels:
        missing_path = os.path.join(args.out_dir, 'missing_labels.txt')
        with open(missing_path, 'w', encoding='utf-8') as f:
            for rel_img_path in missing_labels:
                f.write(rel_img_path + '\n')
        print(f'\nMissing labels written to: {missing_path}')

    print(f'\nBatch visualization results saved to: {args.out_dir}')
    print('Tables:')
    print(f"  {os.path.join(args.out_dir, 'metrics_table.csv')}")
    print(f"  {os.path.join(args.out_dir, 'metrics_table.md')}")
    print(f"  {os.path.join(args.out_dir, 'metrics_long.csv')}")
    print(f"  {os.path.join(args.out_dir, 'metrics_summary.csv')}")


if __name__ == '__main__':
    main()
