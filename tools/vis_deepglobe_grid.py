import argparse
import csv
import os
import re

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize DeepGlobe predictions with config palette and grid.'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='DeepGlobe config file. Also used for data.test.')
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
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='inference device, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--patch-size',
        nargs='+',
        type=int,
        default=None,
        metavar='SIZE',
        help=('grid size. Defaults to config.patch_size, then '
              'config.crop_size'))
    parser.add_argument(
        '--grid-color',
        nargs=3,
        type=int,
        default=(0, 0, 0),
        metavar=('R', 'G', 'B'),
        help='RGB grid line color')
    parser.add_argument(
        '--grid-width',
        type=int,
        default=2,
        help='grid line width in pixels')
    parser.add_argument(
        '--max-num',
        type=int,
        default=None,
        help='visualize only the first N images')
    parser.add_argument(
        '--save-id',
        action='store_true',
        help='also save raw prediction id maps')
    parser.add_argument(
        '--no-compare',
        action='store_true',
        help='do not save side-by-side comparison images')
    parser.add_argument(
        '--table-digits',
        type=int,
        default=4,
        help='decimal digits used in printed and markdown tables')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_patch_size(values):
    if values is None:
        return None
    if isinstance(values, int):
        return (values, values)
    values = list(values)
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) == 2:
        return (values[0], values[1])
    raise ValueError('--patch-size expects one value or two values')


def normalize_rgb(values, name):
    if len(values) != 3:
        raise ValueError(f'{name} expects exactly three RGB values')
    color = tuple(int(v) for v in values)
    if any(v < 0 or v > 255 for v in color):
        raise ValueError(f'{name} RGB values must be in [0, 255]')
    return color


def safe_name(path):
    root = os.path.splitext(path)[0]
    root = root.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', root)


def safe_metric_name(name):
    name = str(name)
    name = re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_')
    return name or 'class'


def build_model_specs(args):
    return [
        ('real', 'real', args.real_config or args.config,
         args.real_checkpoint),
        ('pseudo', 'pseudo', args.pseudo_config or args.config,
         args.pseudo_checkpoint),
        ('patch', 'patch', args.patch_config or args.config,
         args.patch_checkpoint),
    ]


def import_runtime():
    import mmcv

    from mmseg.apis import inference_segmentor, init_segmentor
    from mmseg.datasets import build_dataset

    return mmcv, init_segmentor, inference_segmentor, build_dataset


def iter_leaf_datasets(dataset):
    if hasattr(dataset, 'datasets'):
        for child in dataset.datasets:
            yield from iter_leaf_datasets(child)
        return
    if hasattr(dataset, 'dataset'):
        yield from iter_leaf_datasets(dataset.dataset)
        return
    yield dataset


def build_dataset_from_config(config):
    mmcv, _, _, build_dataset = import_runtime()
    cfg = mmcv.Config.fromfile(config)
    if not hasattr(cfg, 'data') or 'test' not in cfg.data:
        raise KeyError(f'Cannot find data.test in config: {config}')
    dataset = build_dataset(cfg.data.test)
    return cfg, dataset


def infer_patch_size(cfg, args):
    patch_size = normalize_patch_size(args.patch_size)
    if patch_size is not None:
        return patch_size

    if 'patch_size' in cfg:
        return normalize_patch_size(cfg.patch_size)
    if 'crop_size' in cfg:
        return normalize_patch_size(cfg.crop_size)
    return None


def collect_images(dataset):
    images = []
    for leaf in iter_leaf_datasets(dataset):
        img_dir = getattr(leaf, 'img_dir', None)
        ann_dir = getattr(leaf, 'ann_dir', None)
        img_infos = getattr(leaf, 'img_infos', None)
        if img_dir is None or img_infos is None:
            raise ValueError('The config dataset must provide img_dir and '
                             'img_infos.')
        for info in img_infos:
            filename = info['filename']
            ann = info.get('ann', {})
            seg_map = ann.get('seg_map') if ann is not None else None
            label_path = (os.path.join(ann_dir, seg_map)
                          if ann_dir is not None and seg_map is not None else
                          None)
            images.append((os.path.join(img_dir, filename), label_path,
                           filename))
    return images


def get_palette(dataset, model):
    palette = getattr(dataset, 'PALETTE', None)
    classes = getattr(dataset, 'CLASSES', None)
    if palette is None:
        palette = getattr(model, 'PALETTE', None)
    if classes is None:
        classes = getattr(model, 'CLASSES', None)

    if palette is None:
        raise ValueError('Cannot find PALETTE from dataset or checkpoint.')

    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError('PALETTE must have shape [num_classes, 3].')
    return palette, classes


def colorize_prediction(seg, palette):
    seg = np.asarray(seg)
    if seg.ndim != 2:
        raise ValueError(f'Prediction must be a 2D map, got shape {seg.shape}')

    color = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(palette):
        color[seg == class_id] = rgb

    unknown = seg >= len(palette)
    if np.any(unknown):
        color[unknown] = np.array([255, 0, 0], dtype=np.uint8)
    return color


def rgb_label_to_class(rgb, palette):
    label = np.zeros(rgb.shape[:2], dtype=np.uint8)
    matched = np.zeros(rgb.shape[:2], dtype=bool)
    for class_id, color in enumerate(palette):
        mask = np.all(rgb == color, axis=-1)
        label[mask] = class_id
        matched |= mask

    if not matched.all():
        # DeepGlobe configs here usually point at rgb2id labels. This fallback
        # keeps colored _mask.png files usable too.
        label[~matched] = np.any(rgb[~matched] != 0, axis=-1).astype(np.uint8)
    return label


def read_groundtruth(label_path, palette):
    if label_path is None:
        return None
    gt = np.asarray(Image.open(label_path))
    if gt.ndim == 3:
        rgb = gt[:, :, :3]
        if np.all(rgb[:, :, 0] == rgb[:, :, 1]) and np.all(
                rgb[:, :, 0] == rgb[:, :, 2]):
            gt = rgb[:, :, 0]
        else:
            gt = rgb_label_to_class(rgb, palette)
    return gt.astype(np.uint8)


def resize_label(label, target_shape):
    if label.shape == target_shape:
        return label
    target_h, target_w = target_shape
    image = Image.fromarray(label.astype(np.uint8), mode='L')
    image = image.resize((target_w, target_h), Image.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def make_wrong_map(seg, gt):
    wrong = seg != gt
    rgb = np.full((*gt.shape, 3), 255, dtype=np.uint8)
    rgb[wrong] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def class_names_from_palette(classes, palette):
    if classes is None:
        return [f'class_{i}' for i in range(len(palette))]
    return [str(name) for name in classes]


def metric_names(class_names):
    names = ['aAcc', 'mIoU', 'mAcc', 'mDice', 'total', 'correct']
    for class_name in class_names:
        suffix = safe_metric_name(class_name)
        names.extend(
            [f'IoU_{suffix}', f'Acc_{suffix}', f'Dice_{suffix}'])
    return names


def nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return float('nan')
    return float(np.nanmean(values))


def divide_or_nan(numerator, denominator):
    if denominator == 0:
        return float('nan')
    return float(numerator / denominator)


def compute_multiclass_metrics(pred, gt, class_names, ignore_index=255):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f'Prediction shape {pred.shape} does not match '
                         f'GT shape {gt.shape}')

    num_classes = len(class_names)
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    total = int(valid.sum())
    correct = int(((pred == gt) & valid).sum())

    ious = []
    accs = []
    dices = []
    metrics = {
        'total': total,
        'correct': correct,
        'aAcc': divide_or_nan(correct, total),
    }

    for class_id, class_name in enumerate(class_names):
        suffix = safe_metric_name(class_name)
        pred_c = (pred == class_id) & valid
        gt_c = (gt == class_id) & valid
        tp = int((pred_c & gt_c).sum())
        pred_count = int(pred_c.sum())
        gt_count = int(gt_c.sum())
        union = int((pred_c | gt_c).sum())

        iou = divide_or_nan(tp, union)
        acc = divide_or_nan(tp, gt_count)
        dice = divide_or_nan(2 * tp, pred_count + gt_count)
        metrics[f'IoU_{suffix}'] = iou
        metrics[f'Acc_{suffix}'] = acc
        metrics[f'Dice_{suffix}'] = dice
        ious.append(iou)
        accs.append(acc)
        dices.append(dice)

    metrics['mIoU'] = nanmean(ious)
    metrics['mAcc'] = nanmean(accs)
    metrics['mDice'] = nanmean(dices)
    return metrics


def draw_grid(rgb, patch_size, grid_color, grid_width):
    image = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    if patch_size is None:
        return image

    patch_h, patch_w = patch_size
    if patch_h <= 0 or patch_w <= 0 or grid_width <= 0:
        return image

    draw = ImageDraw.Draw(image)
    width, height = image.size
    for x in range(patch_w, width, patch_w):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=grid_width)
    for y in range(patch_h, height, patch_h):
        draw.line([(0, y), (width, y)], fill=grid_color, width=grid_width)
    return image


def make_compare_image(panels, names):
    if len(panels) == 0:
        raise ValueError('No panels to compare.')

    label_h = 32
    pad = 8
    widths = [panel.size[0] for panel in panels]
    heights = [panel.size[1] for panel in panels]
    total_w = sum(widths) + pad * (len(panels) - 1)
    total_h = max(heights) + label_h
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for panel, name in zip(panels, names):
        draw.text((x + 6, 8), name, fill=(0, 0, 0))
        canvas.paste(panel, (x, label_h))
        x += panel.size[0] + pad
    return canvas


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_to_string(value, digits):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ''
        return f'{float(value):.{digits}f}'
    return str(value)


def markdown_table_lines(rows, headers, digits):
    formatted_rows = [[metric_to_string(row.get(header, ''), digits)
                       for header in headers] for row in rows]
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


def write_markdown_table(path, rows, headers, digits):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_table_lines(rows, headers, digits)) + '\n')


def build_wide_row(rel_name, label_path, image_out_dir, per_model_metrics):
    row = {
        'image': rel_name,
        'label': label_path or '',
        'visual_dir': image_out_dir,
    }
    for model_name, metrics in per_model_metrics.items():
        for metric_name, value in metrics.items():
            row[f'{model_name}_{metric_name}'] = value
    return row


def build_long_rows(rel_name, label_path, image_out_dir, per_model_metrics):
    rows = []
    for model_name, metrics in per_model_metrics.items():
        row = {
            'image': rel_name,
            'label': label_path or '',
            'model': model_name,
            'visual_dir': image_out_dir,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def wide_fieldnames(model_names, metrics):
    fields = ['image', 'label', 'visual_dir']
    for model_name in model_names:
        for metric_name in metrics:
            fields.append(f'{model_name}_{metric_name}')
    return fields


def summary_rows(long_rows, model_names, metrics):
    rows = []
    for model_name in model_names:
        model_rows = [row for row in long_rows if row['model'] == model_name]
        if len(model_rows) == 0:
            continue
        summary = {'model': model_name, 'num_images': len(model_rows)}
        for metric_name in metrics:
            values = [row[metric_name] for row in model_rows]
            if metric_name in ('total', 'correct'):
                summary[metric_name] = int(np.nansum(values))
            else:
                summary[metric_name] = nanmean(values)
        rows.append(summary)
    return rows


def compact_rows(wide_rows, model_names):
    rows = []
    keep_metrics = ('aAcc', 'mIoU', 'mAcc', 'mDice')
    for row in wide_rows:
        compact = {'image': row['image']}
        for model_name in model_names:
            for metric_name in keep_metrics:
                compact[f'{model_name}_{metric_name}'] = row.get(
                    f'{model_name}_{metric_name}', '')
        rows.append(compact)
    return rows


def main():
    args = parse_args()
    args.patch_size = normalize_patch_size(args.patch_size)
    args.grid_color = normalize_rgb(args.grid_color, '--grid-color')
    model_specs = build_model_specs(args)
    if args.grid_width < 0:
        raise ValueError('--grid-width must be >= 0')

    ensure_dir(args.out_dir)
    per_image_root = os.path.join(args.out_dir, 'per_image')
    ensure_dir(per_image_root)

    mmcv, init_segmentor, inference_segmentor, _ = import_runtime()
    cfg, dataset = build_dataset_from_config(args.config)
    models = []
    for name, safe_model_name, config, checkpoint in model_specs:
        model = init_segmentor(config, checkpoint, device=args.device)
        models.append((name, safe_model_name, model))

    palette, classes = get_palette(dataset, models[0][2])
    class_names = class_names_from_palette(classes, palette)
    all_metric_names = metric_names(class_names)
    model_names = [name for name, _, _ in models]
    patch_size = infer_patch_size(cfg, args)
    images = collect_images(dataset)
    if args.max_num is not None:
        images = images[:args.max_num]
    if len(images) == 0:
        raise FileNotFoundError('No images found from config.data.test.')

    print(f'Loaded {len(images)} DeepGlobe images')
    print(f'Palette classes: {classes}')
    print(f'Grid size: {patch_size}')
    print('Models:')
    for name, _, _, checkpoint in model_specs:
        print(f'  {name}: {checkpoint}')

    wide_rows = []
    long_rows = []

    for index, (img_path, label_path, rel_name) in enumerate(images, start=1):
        image_out_dir = os.path.join(per_image_root, safe_name(rel_name))
        ensure_dir(image_out_dir)
        compare_panels = []
        compare_names = []
        per_model_metrics = {}
        gt = read_groundtruth(label_path, palette)
        if gt is not None:
            gt_color = colorize_prediction(gt, palette)
            gt_grid = draw_grid(gt_color, patch_size, args.grid_color,
                                args.grid_width)
            gt_grid.save(
                os.path.join(image_out_dir, 'groundtruth_color_grid.png'))
        else:
            print(f'[{index}/{len(images)}] missing GT for {rel_name}')

        for model_name, safe_model_name, model in models:
            result = inference_segmentor(model, img_path)
            seg = result[0]
            color = colorize_prediction(seg, palette)
            color_grid = draw_grid(color, patch_size, args.grid_color,
                                   args.grid_width)
            color_grid.save(
                os.path.join(image_out_dir,
                             f'{safe_model_name}_pred_color_grid.png'))

            if args.save_id:
                id_path = os.path.join(image_out_dir,
                                       f'{safe_model_name}_pred_id.png')
                Image.fromarray(
                    np.asarray(seg, dtype=np.uint8), mode='L').save(id_path)

            if gt is not None:
                gt_for_pred = resize_label(gt, seg.shape)
                wrong = make_wrong_map(seg, gt_for_pred)
                wrong_grid = draw_grid(wrong, patch_size, args.grid_color,
                                       args.grid_width)
                wrong_grid.save(
                    os.path.join(image_out_dir,
                                 f'{safe_model_name}_wrong_grid.png'))
                per_model_metrics[model_name] = compute_multiclass_metrics(
                    seg, gt_for_pred, class_names)

            compare_panels.append(color_grid)
            compare_names.append(model_name)

        if not args.no_compare and len(compare_panels) > 1:
            compare_image = make_compare_image(compare_panels, compare_names)
            compare_image.save(os.path.join(image_out_dir, 'compare_grid.png'))

        if per_model_metrics:
            wide_rows.append(
                build_wide_row(rel_name, label_path, image_out_dir,
                               per_model_metrics))
            long_rows.extend(
                build_long_rows(rel_name, label_path, image_out_dir,
                                per_model_metrics))

        print(f'[{index}/{len(images)}] saved {image_out_dir}')

    print(f'DeepGlobe grid visualizations saved to: {per_image_root}')

    if wide_rows:
        wide_fields = wide_fieldnames(model_names, all_metric_names)
        long_fields = ['image', 'label', 'model',
                       'visual_dir'] + all_metric_names
        summary_fields = ['model', 'num_images'] + all_metric_names

        metrics_table = os.path.join(args.out_dir, 'metrics_table.csv')
        metrics_long = os.path.join(args.out_dir, 'metrics_long.csv')
        metrics_summary = os.path.join(args.out_dir, 'metrics_summary.csv')
        metrics_md = os.path.join(args.out_dir, 'metrics_table.md')

        write_csv(metrics_table, wide_rows, wide_fields)
        write_csv(metrics_long, long_rows, long_fields)
        write_csv(metrics_summary,
                  summary_rows(long_rows, model_names, all_metric_names),
                  summary_fields)

        compact = compact_rows(wide_rows, model_names)
        compact_headers = list(compact[0].keys())
        write_markdown_table(metrics_md, compact, compact_headers,
                             args.table_digits)

        print('Metric tables:')
        print(f'  {metrics_table}')
        print(f'  {metrics_md}')
        print(f'  {metrics_long}')
        print(f'  {metrics_summary}')
    else:
        print('No metric tables written because no GT labels were found.')


if __name__ == '__main__':
    main()
