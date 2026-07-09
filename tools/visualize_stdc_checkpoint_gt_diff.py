import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class Prediction:
    name: str
    pred: np.ndarray
    fg_prob: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize three STDC checkpoints and GT differences.')
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
        '--img',
        required=True,
        help='same input image used by all three checkpoints')
    parser.add_argument(
        '--gt',
        required=True,
        help='binary ground-truth mask for this image, 0=background, >0=foreground'
    )
    parser.add_argument('--out-dir', required=True)

    parser.add_argument(
        '--device',
        default='cuda:0',
        help='inference device, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--bg-class', type=int, default=0, help='background class index')
    parser.add_argument(
        '--fg-class', type=int, default=1, help='foreground class index')
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
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def normalize_colors(args):
    args.over_color = normalize_rgb(args.over_color, '--over-color')
    args.under_color = normalize_rgb(args.under_color, '--under-color')
    args.true_fg_color = normalize_rgb(args.true_fg_color, '--true-fg-color')
    args.true_bg_color = normalize_rgb(args.true_bg_color, '--true-bg-color')


def import_mmseg_runtime():
    import mmcv
    import torch
    from mmcv.parallel import collate, scatter

    from mmseg.apis.inference import LoadImage, init_segmentor
    from mmseg.datasets.pipelines import Compose

    return mmcv, torch, collate, scatter, LoadImage, init_segmentor, Compose


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


def read_groundtruth(path):
    gt = np.asarray(Image.open(path))
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return (gt > 0).astype(np.uint8)


def checkpoint_to_prediction(name, config, checkpoint, img_path, args):
    model = init_model(config, checkpoint, args.device)
    prob = infer_prob(model, img_path)
    if args.bg_class >= prob.shape[0] or args.fg_class >= prob.shape[0]:
        raise ValueError(
            f'{name} output has {prob.shape[0]} classes, but bg_class='
            f'{args.bg_class} and fg_class={args.fg_class}')

    pred = prob.argmax(axis=0).astype(np.uint8)
    fg_prob = prob[args.fg_class].astype(np.float32)
    return Prediction(name, pred, fg_prob)


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


def save_metrics(metrics, out_dir):
    metric_names = [
        'aAcc', 'mIoU', 'mAcc', 'IoU_background', 'IoU_foreground',
        'Acc_background', 'Acc_foreground', 'Dice_foreground', 'TP', 'TN',
        'FP', 'FN'
    ]
    csv_path = os.path.join(out_dir, 'metrics.csv')
    txt_path = os.path.join(out_dir, 'metrics.txt')

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('name,' + ','.join(metric_names) + '\n')
        for name, values in metrics.items():
            row = [name]
            for metric_name in metric_names:
                value = values[metric_name]
                if isinstance(value, int):
                    row.append(str(value))
                else:
                    row.append(f'{value:.6f}')
            f.write(','.join(row) + '\n')

    with open(txt_path, 'w', encoding='utf-8') as f:
        for name, values in metrics.items():
            f.write(f'[{name}]\n')
            for metric_name in metric_names:
                value = values[metric_name]
                if isinstance(value, int):
                    f.write(f'{metric_name}: {value}\n')
                else:
                    f.write(f'{metric_name}: {value:.6f}\n')
            f.write('\n')


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
            fg_like = (pred == args.fg_class).astype(np.float32)
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


def main():
    args = parse_args()
    normalize_patch_size(args)
    normalize_colors(args)
    ensure_dir(args.out_dir)

    real_config = args.real_config or args.config
    pseudo_config = args.pseudo_config or args.config
    patch_config = args.patch_config or args.config

    image_rgb = read_image_rgb(args.img)
    gt = read_groundtruth(args.gt)

    save_rgb_image(image_rgb, os.path.join(args.out_dir, 'image.png'))
    if args.patch_size is not None:
        save_image_with_grid(
            image_rgb,
            os.path.join(args.out_dir, 'image_patch_grid.png'),
            args,
            title='image')

    real = checkpoint_to_prediction('real', real_config,
                                    args.real_checkpoint, args.img, args)
    pseudo = checkpoint_to_prediction('pseudo', pseudo_config,
                                      args.pseudo_checkpoint, args.img, args)
    patch = checkpoint_to_prediction('patch', patch_config,
                                     args.patch_checkpoint, args.img, args)
    predictions = (real, pseudo, patch)

    for prediction in predictions:
        save_prediction_outputs(prediction, args.out_dir, args)

    gt_path = os.path.join(args.out_dir, 'groundtruth.png')
    save_binary_mask(gt, gt_path)
    if args.patch_size is not None:
        save_mask_with_grid(
            gt,
            os.path.join(args.out_dir, 'groundtruth_patch_grid.png'),
            args,
            title='ground truth')

    aligned_predictions = {}
    diff_maps = {}
    metrics = {}
    for prediction in predictions:
        box = prediction_box(args, prediction.name)
        pred_for_gt = align_prediction_to_gt(prediction.pred, gt.shape, box)
        aligned_predictions[prediction.name] = pred_for_gt

        if box is not None or pred_for_gt.shape != prediction.pred.shape:
            save_pred_map(
                pred_for_gt,
                os.path.join(args.out_dir,
                             f'{prediction.name}_pred_for_gt.png'))

        diff_map = build_gt_diff_map(pred_for_gt, gt, args)
        diff_maps[prediction.name] = diff_map
        save_rgb_image(
            diff_map,
            os.path.join(args.out_dir, f'{prediction.name}_gt_diff.png'))
        if args.patch_size is not None:
            save_image_with_grid(
                diff_map,
                os.path.join(args.out_dir,
                             f'{prediction.name}_gt_diff_patch_grid.png'),
                args,
                title=f'{prediction.name} vs GT')

        metrics[prediction.name] = compute_binary_metrics(
            pred_for_gt, gt, args)

    save_metrics(metrics, args.out_dir)
    save_summary(image_rgb, gt, aligned_predictions, diff_maps, metrics,
                 os.path.join(args.out_dir, 'summary.png'), args)

    print(f'Visualization results saved to: {args.out_dir}')
    print('Metrics:')
    for name, values in metrics.items():
        print(
            f"{name}: aAcc={values['aAcc']:.4f}, "
            f"mIoU={values['mIoU']:.4f}, "
            f"IoU_fg={values['IoU_foreground']:.4f}, "
            f"Dice_fg={values['Dice_foreground']:.4f}, "
            f"over(FP)={values['FP']}, under(FN)={values['FN']}")


if __name__ == '__main__':
    main()
