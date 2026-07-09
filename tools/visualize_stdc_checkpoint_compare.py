import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class Prediction:
    name: str
    pred: np.ndarray
    bg_prob: np.ndarray
    fg_prob: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize STDC predictions from three checkpoints.')
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
        help='binary ground-truth mask for this image, 0=background, 1=foreground')
    parser.add_argument(
        '--out-dir', required=True, help='directory to save visualizations')

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
        help='crop real-checkpoint prediction before comparing with patch')
    parser.add_argument(
        '--pseudo-box',
        nargs=4,
        type=int,
        default=None,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='crop pseudo-checkpoint prediction before comparing with patch')
    parser.add_argument(
        '--cmap', default='magma', help='matplotlib colormap for heatmaps')
    parser.add_argument(
        '--dpi', type=int, default=200, help='saved figure dpi')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.45,
        help='overlay opacity for foreground probability on the input image')
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

    # Keep torch imported in this function for environments that lazily load
    # CUDA context; this also makes linters aware the dependency is intentional.
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
            prob += model.inference(imgs[aug_idx], img_metas[aug_idx],
                                    rescale=True)
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
    bg_prob = prob[args.bg_class].astype(np.float32)
    fg_prob = prob[args.fg_class].astype(np.float32)
    return Prediction(name, pred, bg_prob, fg_prob)


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


def save_heatmap(data,
                 path,
                 title,
                 cmap,
                 dpi,
                 args=None,
                 vmin=0.0,
                 vmax=1.0):
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


def save_overlay(image_rgb, heatmap, path, title, cmap, dpi, opacity, args):
    if image_rgb.shape[:2] != heatmap.shape:
        image_rgb = resize_rgb(image_rgb, heatmap.shape)

    fig, ax = plt.subplots(figsize=(max(heatmap.shape[1] / dpi, 3.0),
                                    max(heatmap.shape[0] / dpi, 3.0)),
                           dpi=dpi)
    ax.imshow(image_rgb)
    ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1, alpha=opacity)
    ax.set_title(title)
    ax.axis('off')
    draw_patch_grid(ax, heatmap.shape, args)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()


def resize_rgb(image, target_shape):
    target_h, target_w = target_shape
    pil_img = Image.fromarray(image.astype(np.uint8))
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(pil_img)


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


def resize_label(data, target_shape):
    target_h, target_w = target_shape
    if data.shape == target_shape:
        return data
    image = Image.fromarray(data.astype(np.uint8), mode='L')
    image = image.resize((target_w, target_h), Image.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def residual(left, right):
    if left.shape != right.shape:
        right = resize_map(right, left.shape)
    return np.abs(left.astype(np.float32) - right.astype(np.float32))


def compute_binary_metrics(pred, gt):
    if pred.shape != gt.shape:
        pred = resize_label(pred, gt.shape)

    pred = (pred > 0).astype(np.uint8)
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


def save_prediction_outputs(prediction, image_path, out_dir, args):
    pred_path = os.path.join(out_dir, f'{prediction.name}_pred.png')
    save_pred_map(prediction.pred, pred_path)
    if args.patch_size is not None:
        grid_path = os.path.join(
            out_dir, f'{prediction.name}_pred_patch_grid.png')
        save_mask_with_grid(prediction.pred, grid_path, args)
    save_heatmap(
        prediction.fg_prob,
        os.path.join(out_dir, f'{prediction.name}_foreground_heatmap.png'),
        f'{prediction.name} foreground',
        args.cmap,
        args.dpi,
        args=args)
    save_heatmap(
        prediction.bg_prob,
        os.path.join(out_dir, f'{prediction.name}_background_heatmap.png'),
        f'{prediction.name} background',
        args.cmap,
        args.dpi,
        args=args)
    save_overlay(
        read_image_rgb(image_path),
        prediction.fg_prob,
        os.path.join(out_dir, f'{prediction.name}_foreground_overlay.png'),
        f'{prediction.name} foreground overlay',
        args.cmap,
        args.dpi,
        args.opacity,
        args)


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


def save_summary(predictions, residuals, gt, metrics, out_path, args):
    rows = len(predictions) + 2
    cols = 3
    plt.figure(figsize=(cols * 4, rows * 4), dpi=args.dpi)

    plt.subplot(rows, cols, 1)
    ax = plt.gca()
    ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
    ax.set_title('ground truth')
    ax.axis('off')
    draw_patch_grid(ax, gt.shape, args)

    plt.subplot(rows, cols, 2)
    ax = plt.gca()
    ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
    ax.set_title('GT foreground')
    ax.axis('off')
    draw_patch_grid(ax, gt.shape, args)

    plt.subplot(rows, cols, 3)
    ax = plt.gca()
    ax.imshow(1 - gt, cmap='gray', vmin=0, vmax=1)
    ax.set_title('GT background')
    ax.axis('off')
    draw_patch_grid(ax, gt.shape, args)

    for row, prediction in enumerate(predictions):
        offset = (row + 1) * cols
        title_suffix = ''
        if prediction.name in metrics:
            title_suffix = (
                f" aAcc={metrics[prediction.name]['aAcc']:.3f}, "
                f"mIoU={metrics[prediction.name]['mIoU']:.3f}")

        plt.subplot(rows, cols, offset + 1)
        ax = plt.gca()
        ax.imshow(prediction.pred, cmap='gray')
        ax.set_title(f'{prediction.name} pred{title_suffix}')
        ax.axis('off')
        draw_patch_grid(ax, prediction.pred.shape, args)

        plt.subplot(rows, cols, offset + 2)
        ax = plt.gca()
        ax.imshow(prediction.fg_prob, cmap=args.cmap, vmin=0, vmax=1)
        ax.set_title(f'{prediction.name} foreground')
        ax.axis('off')
        draw_patch_grid(ax, prediction.fg_prob.shape, args)

        plt.subplot(rows, cols, offset + 3)
        ax = plt.gca()
        ax.imshow(prediction.bg_prob, cmap=args.cmap, vmin=0, vmax=1)
        ax.set_title(f'{prediction.name} background')
        ax.axis('off')
        draw_patch_grid(ax, prediction.bg_prob.shape, args)

    for col, (name, value) in enumerate(residuals.items()):
        plt.subplot(rows, cols, (len(predictions) + 1) * cols + col + 1)
        ax = plt.gca()
        ax.imshow(value, cmap=args.cmap, vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis('off')
        draw_patch_grid(ax, value.shape, args)

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    normalize_patch_size(args)
    ensure_dir(args.out_dir)

    real_config = args.real_config or args.config
    pseudo_config = args.pseudo_config or args.config
    patch_config = args.patch_config or args.config

    real = checkpoint_to_prediction('real', real_config,
                                    args.real_checkpoint, args.img, args)
    pseudo = checkpoint_to_prediction('pseudo', pseudo_config,
                                      args.pseudo_checkpoint,
                                      args.img, args)
    patch = checkpoint_to_prediction('patch', patch_config,
                                     args.patch_checkpoint, args.img, args)
    gt = read_groundtruth(args.gt)

    save_prediction_outputs(real, args.img, args.out_dir, args)
    save_prediction_outputs(pseudo, args.img, args.out_dir, args)
    save_prediction_outputs(patch, args.img, args.out_dir, args)
    gt_path = os.path.join(args.out_dir, 'groundtruth.png')
    save_binary_mask(gt, gt_path)
    if args.patch_size is not None:
        save_mask_with_grid(
            gt,
            os.path.join(args.out_dir, 'groundtruth_patch_grid.png'),
            args,
            title='ground truth')

    metrics = {
        prediction.name: compute_binary_metrics(prediction.pred, gt)
        for prediction in (real, pseudo, patch)
    }
    save_metrics(metrics, args.out_dir)

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
        save_heatmap(value,
                     os.path.join(args.out_dir, f'{name}.png'),
                     name,
                     args.cmap,
                     args.dpi,
                     args=args,
                     vmin=0.0,
                     vmax=1.0)

    save_summary((real, pseudo, patch), residuals, gt, metrics,
                 os.path.join(args.out_dir, 'summary.png'), args)

    print(f'Visualization results saved to: {args.out_dir}')
    print('Metrics:')
    for name, values in metrics.items():
        print(
            f"{name}: aAcc={values['aAcc']:.4f}, "
            f"mIoU={values['mIoU']:.4f}, "
            f"IoU_fg={values['IoU_foreground']:.4f}, "
            f"Dice_fg={values['Dice_foreground']:.4f}")


if __name__ == '__main__':
    main()
