import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
DIFFICULTIES = ('easy', 'medium', 'hard')


@dataclass
class PatchStats:
    image_path: Path
    label_path: Path
    class_counts: dict
    valid_pixels: int
    boundary_count: int
    valid_boundary_pairs: int
    boundary_density: float
    entropy: float = 0.0
    dominant_ratio: float = 0.0
    non_dominant: float = 0.0
    boundary_percentile: float = 0.0
    difficulty_score: float = 0.0
    difficulty: str = ''


def is_image_file(path):
    return path.suffix.lower() in IMG_EXTS


def collect_pairs(img_dir, label_dir, match_by_stem=False):
    img_paths = sorted(
        p for p in img_dir.iterdir() if p.is_file() and is_image_file(p))
    label_paths = sorted(
        p for p in label_dir.iterdir() if p.is_file() and is_image_file(p))

    if match_by_stem:
        label_map = {p.stem: p for p in label_paths}
        image_key = lambda p: p.stem
    else:
        label_map = {p.name: p for p in label_paths}
        image_key = lambda p: p.name

    pairs = []
    missing = []
    used_label_keys = set()
    for img_path in img_paths:
        key = image_key(img_path)
        label_path = label_map.get(key)
        if label_path is None:
            missing.append(img_path.name)
            continue

        pairs.append((img_path, label_path))
        used_label_keys.add(key)

    if missing:
        preview = ', '.join(missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} image patches do not have matching labels. '
            f'First missing: {preview}')

    extra = sorted(set(label_map) - used_label_keys)
    if extra:
        preview = ', '.join(extra[:10])
        raise FileNotFoundError(
            f'{len(extra)} label patches do not have matching images. '
            f'First extra: {preview}')

    return pairs


def read_label_array(label_path):
    with Image.open(label_path) as img:
        arr = np.array(img)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3].astype(np.uint32)
        return (arr[:, :, 0] << 16) + (arr[:, :, 1] << 8) + arr[:, :, 2]

    raise ValueError(f'Unsupported label shape {arr.shape}: {label_path}')


def check_patch_size(img_path, label_path, expected_size):
    if expected_size <= 0:
        return

    with Image.open(img_path) as img:
        img_size = img.size
    with Image.open(label_path) as label:
        label_size = label.size

    expected = (expected_size, expected_size)
    if img_size != expected:
        raise ValueError(f'Image is not {expected}: {img_path} {img_size}')
    if label_size != expected:
        raise ValueError(f'Label is not {expected}: {label_path} {label_size}')


def make_valid_mask(label, ignore_labels):
    if ignore_labels is None or len(ignore_labels) == 0:
        return np.ones(label.shape, dtype=bool)

    return ~np.isin(label, np.array(ignore_labels, dtype=label.dtype))


def compute_boundary_density(label, valid):
    right_valid = valid[:, :-1] & valid[:, 1:]
    down_valid = valid[:-1, :] & valid[1:, :]

    right_boundary = (label[:, :-1] != label[:, 1:]) & right_valid
    down_boundary = (label[:-1, :] != label[1:, :]) & down_valid

    boundary_count = int(right_boundary.sum() + down_boundary.sum())
    valid_pairs = int(right_valid.sum() + down_valid.sum())
    density = boundary_count / valid_pairs if valid_pairs > 0 else 0.0

    return boundary_count, valid_pairs, density


def compute_basic_stats(img_path, label_path, ignore_labels, patch_size):
    check_patch_size(img_path, label_path, patch_size)

    label = read_label_array(label_path)
    valid = make_valid_mask(label, ignore_labels)
    valid_values = label[valid]
    valid_pixels = int(valid_values.size)

    if valid_pixels == 0:
        raise ValueError(f'No valid pixels in label: {label_path}')

    values, counts = np.unique(valid_values, return_counts=True)
    class_counts = {
        int(value): int(count)
        for value, count in zip(values, counts)
    }
    boundary_count, valid_pairs, boundary_density = compute_boundary_density(
        label, valid)

    return PatchStats(
        image_path=img_path,
        label_path=label_path,
        class_counts=class_counts,
        valid_pixels=valid_pixels,
        boundary_count=boundary_count,
        valid_boundary_pairs=valid_pairs,
        boundary_density=boundary_density)


def percentile_rank(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([0.0], dtype=np.float64)

    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1

        avg_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg_rank / (n - 1)
        start = end

    return ranks


def fill_scores(stats_list, num_classes=None):
    if num_classes is None:
        class_values = set()
        for stats in stats_list:
            class_values.update(stats.class_counts)
        num_classes = len(class_values)

    if num_classes < 1:
        raise ValueError('num_classes must be at least 1.')

    b_values = [stats.boundary_density for stats in stats_list]
    b_percentiles = percentile_rank(b_values)
    entropy_denominator = np.log(num_classes) if num_classes > 1 else 0.0

    for stats, b_hat in zip(stats_list, b_percentiles):
        probs = np.array(
            list(stats.class_counts.values()), dtype=np.float64)
        probs = probs / stats.valid_pixels

        if entropy_denominator > 0:
            entropy = float(-(probs * np.log(probs)).sum() /
                            entropy_denominator)
        else:
            entropy = 0.0

        dominant_ratio = float(probs.max())
        non_dominant = 1.0 - dominant_ratio
        difficulty_score = 0.5 * float(b_hat) + 0.3 * entropy + \
            0.2 * non_dominant

        stats.entropy = entropy
        stats.dominant_ratio = dominant_ratio
        stats.non_dominant = non_dominant
        stats.boundary_percentile = float(b_hat)
        stats.difficulty_score = difficulty_score


def assign_difficulty(stats_list):
    sorted_stats = sorted(stats_list, key=lambda x: x.difficulty_score)
    n = len(sorted_stats)
    easy_end = n // 3
    medium_end = (2 * n) // 3

    for idx, stats in enumerate(sorted_stats):
        if idx < easy_end:
            stats.difficulty = 'easy'
        elif idx < medium_end:
            stats.difficulty = 'medium'
        else:
            stats.difficulty = 'hard'


def copy_or_move(src, dst, move=False, overwrite=False, dry_run=False):
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f'Output file already exists: {dst}. Use --overwrite to '
                f'replace it.')
        if dst.is_file():
            dst.unlink()

    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def write_outputs(stats_list, out_root, img_subdir, label_subdir, move=False,
                  overwrite=False, dry_run=False):
    for stats in stats_list:
        out_img = out_root / stats.difficulty / img_subdir / \
            stats.image_path.name
        out_label = out_root / stats.difficulty / label_subdir / \
            stats.label_path.name
        copy_or_move(
            stats.image_path, out_img, move=move, overwrite=overwrite,
            dry_run=dry_run)
        copy_or_move(
            stats.label_path, out_label, move=move, overwrite=overwrite,
            dry_run=dry_run)


def write_csv(stats_list, csv_path, dry_run=False):
    if dry_run:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'image', 'label', 'difficulty', 'difficulty_score',
        'boundary_density', 'boundary_percentile', 'entropy',
        'dominant_ratio', 'non_dominant', 'valid_pixels',
        'boundary_count', 'valid_boundary_pairs', 'class_counts'
    ]

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stats in sorted(stats_list, key=lambda x: x.difficulty_score):
            writer.writerow({
                'image': stats.image_path.name,
                'label': stats.label_path.name,
                'difficulty': stats.difficulty,
                'difficulty_score': f'{stats.difficulty_score:.8f}',
                'boundary_density': f'{stats.boundary_density:.8f}',
                'boundary_percentile': f'{stats.boundary_percentile:.8f}',
                'entropy': f'{stats.entropy:.8f}',
                'dominant_ratio': f'{stats.dominant_ratio:.8f}',
                'non_dominant': f'{stats.non_dominant:.8f}',
                'valid_pixels': stats.valid_pixels,
                'boundary_count': stats.boundary_count,
                'valid_boundary_pairs': stats.valid_boundary_pairs,
                'class_counts': stats.class_counts,
            })


def write_distribution_plot(stats_list, plot_path, dry_run=False):
    if dry_run:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            'matplotlib is required to save the distribution plot. '
            'Install matplotlib or run with --dry_run to skip writing.'
        ) from exc

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        'easy': '#4daf4a',
        'medium': '#377eb8',
        'hard': '#e41a1c',
    }
    by_difficulty = {
        name: [
            stats.difficulty_score for stats in stats_list
            if stats.difficulty == name
        ]
        for name in DIFFICULTIES
    }

    b_hat = [stats.boundary_percentile for stats in stats_list]
    entropy = [stats.entropy for stats in stats_list]
    non_dominant = [stats.non_dominant for stats in stats_list]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Patch Difficulty Distribution', fontsize=14)

    axes[0, 0].hist(
        [by_difficulty[name] for name in DIFFICULTIES],
        bins=40,
        stacked=True,
        color=[colors[name] for name in DIFFICULTIES],
        label=DIFFICULTIES)
    axes[0, 0].set_title('Difficulty Score D')
    axes[0, 0].set_xlabel('D')
    axes[0, 0].set_ylabel('Patch count')
    axes[0, 0].legend()

    axes[0, 1].hist(b_hat, bins=40, color='#984ea3')
    axes[0, 1].set_title('Boundary Percentile B_hat')
    axes[0, 1].set_xlabel('B_hat')
    axes[0, 1].set_ylabel('Patch count')

    axes[1, 0].hist(entropy, bins=40, color='#ff7f00')
    axes[1, 0].set_title('Class Entropy H')
    axes[1, 0].set_xlabel('H')
    axes[1, 0].set_ylabel('Patch count')

    axes[1, 1].hist(non_dominant, bins=40, color='#a65628')
    axes[1, 1].set_title('Non-dominant Ratio A')
    axes[1, 1].set_xlabel('A')
    axes[1, 1].set_ylabel('Patch count')

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def summarize(stats_list):
    counts = {name: 0 for name in DIFFICULTIES}
    for stats in stats_list:
        counts[stats.difficulty] += 1

    scores = np.array([s.difficulty_score for s in stats_list])
    boundaries = np.array([s.boundary_density for s in stats_list])
    entropies = np.array([s.entropy for s in stats_list])

    print('=' * 60)
    print(f'Total patches: {len(stats_list)}')
    print(
        'Split counts: '
        f'easy={counts["easy"]}, medium={counts["medium"]}, '
        f'hard={counts["hard"]}')
    print(
        f'D score: min={scores.min():.6f}, '
        f'mean={scores.mean():.6f}, max={scores.max():.6f}')
    print(
        f'Boundary density: min={boundaries.min():.6f}, '
        f'mean={boundaries.mean():.6f}, max={boundaries.max():.6f}')
    print(
        f'Entropy: min={entropies.min():.6f}, '
        f'mean={entropies.mean():.6f}, max={entropies.max():.6f}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split 500x500 segmentation patches into easy, medium, '
                    'and hard folders by D=0.5*B_hat+0.3*H+0.2*A.')
    parser.add_argument(
        '--data_root',
        type=str,
        default='/root/autodl-tmp/aerial_500',
        help='Dataset root. Default: /root/autodl-tmp/aerial_500')
    parser.add_argument(
        '--out_root',
        type=str,
        default=None,
        help='Output root for easy/medium/hard folders. If omitted, only '
             'the distribution plot is written.')
    parser.add_argument(
        '--plot_path',
        type=str,
        default=None,
        help='Path of the distribution plot. Default: '
             '<out_root>/difficulty_distribution.png when --out_root is set, '
             'otherwise <data_root>/difficulty_distribution.png.')
    parser.add_argument(
        '--img_subdir',
        type=str,
        default='imgs/train',
        help='Image patch subdir under data_root.')
    parser.add_argument(
        '--label_subdir',
        type=str,
        default='labels/train',
        help='Label patch subdir under data_root.')
    parser.add_argument(
        '--patch_size',
        type=int,
        default=500,
        help='Expected square patch size. Use 0 to skip size checking.')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='Class count for entropy normalization. Default: infer from '
             'all valid labels.')
    parser.add_argument(
        '--ignore_labels',
        type=int,
        nargs='*',
        default=None,
        help='Label values to ignore, e.g. --ignore_labels 255. Default: '
             'do not ignore any value.')
    parser.add_argument(
        '--match_by_stem',
        action='store_true',
        help='Match image and label by file stem instead of full filename.')
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying them.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files.')
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only compute scores and print summary; do not write files.')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else None
    if args.plot_path:
        plot_path = Path(args.plot_path)
    elif out_root is not None:
        plot_path = out_root / 'difficulty_distribution.png'
    else:
        plot_path = data_root / 'difficulty_distribution.png'

    if args.move and out_root is None:
        raise ValueError('--move requires --out_root.')

    img_subdir = Path(args.img_subdir)
    label_subdir = Path(args.label_subdir)
    img_dir = data_root / img_subdir
    label_dir = data_root / label_subdir

    if not img_dir.exists():
        raise FileNotFoundError(f'Image dir does not exist: {img_dir}')
    if not label_dir.exists():
        raise FileNotFoundError(f'Label dir does not exist: {label_dir}')

    pairs = collect_pairs(
        img_dir, label_dir, match_by_stem=args.match_by_stem)
    if len(pairs) == 0:
        raise RuntimeError(f'No image/label patch pairs found in {img_dir}')

    print(f'Found {len(pairs)} image/label pairs.')
    stats_list = []
    for idx, (img_path, label_path) in enumerate(pairs, 1):
        stats = compute_basic_stats(
            img_path=img_path,
            label_path=label_path,
            ignore_labels=args.ignore_labels,
            patch_size=args.patch_size)
        stats_list.append(stats)
        if idx % 100 == 0 or idx == len(pairs):
            print(f'Computed stats: {idx}/{len(pairs)}')

    fill_scores(stats_list, num_classes=args.num_classes)
    assign_difficulty(stats_list)
    summarize(stats_list)
    write_distribution_plot(stats_list, plot_path, dry_run=args.dry_run)

    if out_root is not None:
        write_outputs(
            stats_list=stats_list,
            out_root=out_root,
            img_subdir=img_subdir,
            label_subdir=label_subdir,
            move=args.move,
            overwrite=args.overwrite,
            dry_run=args.dry_run)
        csv_path = out_root / 'difficulty_stats.csv'
        write_csv(stats_list, csv_path, dry_run=args.dry_run)

    if args.dry_run:
        print('[DRY RUN] No files were written.')
    elif out_root is None:
        print('No --out_root was set. Skipped folder split and CSV writing.')
        print(f'Distribution plot: {plot_path}')
    else:
        action = 'Moved' if args.move else 'Copied'
        print(f'{action} patches to: {out_root}')
        print(f'Stats CSV: {csv_path}')
        print(f'Distribution plot: {plot_path}')


if __name__ == '__main__':
    main()
