import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_MONUSEG_ROOT = r'E:\dataset\MoNuSeg'
IMAGE_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
SKIP_DIRS = {'__MACOSX'}


@dataclass
class Sample:
    split: str
    xml_path: Path
    image_path: Path


def is_skipped_path(path: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith('._') for part in path.parts)


def infer_split(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if any('monusegtestdata' in part for part in parts):
        return 'test'
    if any('training' in part for part in parts):
        return 'train'
    return 'all'


def candidate_image_paths(xml_path: Path, root: Path):
    stem = xml_path.stem
    search_dirs = [xml_path.parent]

    if xml_path.parent.name.lower() == 'annotations':
        search_dirs.append(xml_path.parent.parent / 'Tissue Images')

    for directory in search_dirs:
        for suffix in IMAGE_SUFFIXES:
            yield directory / f'{stem}{suffix}'

    for suffix in IMAGE_SUFFIXES:
        yield from root.rglob(f'{stem}{suffix}')


def find_image_for_xml(xml_path: Path, root: Path) -> Path:
    seen = set()
    for candidate in candidate_image_paths(xml_path, root):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and not is_skipped_path(candidate):
            return candidate

    raise FileNotFoundError(f'No matching image found for XML: {xml_path}')


def discover_samples(root: Path):
    xml_files = sorted(
        path for path in root.rglob('*.xml')
        if path.is_file() and not is_skipped_path(path)
    )

    samples = []
    for xml_path in xml_files:
        image_path = find_image_for_xml(xml_path, root)
        samples.append(Sample(infer_split(xml_path), xml_path, image_path))

    return samples


def parse_region_vertices(xml_path: Path):
    positive_regions = []
    negative_regions = []

    tree = ET.parse(xml_path)
    for region in tree.iter('Region'):
        vertices_node = region.find('Vertices')
        if vertices_node is None:
            continue

        vertices = []
        for vertex in vertices_node.findall('Vertex'):
            try:
                x = float(vertex.attrib['X'])
                y = float(vertex.attrib['Y'])
            except (KeyError, ValueError):
                continue

            if math.isfinite(x) and math.isfinite(y):
                vertices.append((x, y))

        if len(vertices) < 3:
            continue

        if region.attrib.get('NegativeROA', '0') == '1':
            negative_regions.append(vertices)
        else:
            positive_regions.append(vertices)

    return positive_regions, negative_regions


def clamp_and_round(vertices, width: int, height: int):
    clamped = []
    for x, y in vertices:
        px = min(max(int(round(x)), 0), width - 1)
        py = min(max(int(round(y)), 0), height - 1)
        clamped.append((px, py))
    return clamped


def rasterize_regions(positive_regions, negative_regions, size):
    width, height = size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for vertices in positive_regions:
        draw.polygon(clamp_and_round(vertices, width, height), outline=1, fill=1)

    for vertices in negative_regions:
        draw.polygon(clamp_and_round(vertices, width, height), outline=0, fill=0)

    return mask


def convert_one(sample: Sample, output_path: Path, overwrite: bool, verbose: bool):
    if output_path.exists() and not overwrite:
        if verbose:
            print(f'[SKIP] Exists: {output_path}')
        return False

    with Image.open(sample.image_path) as image:
        size = image.size

    positive_regions, negative_regions = parse_region_vertices(sample.xml_path)
    mask = rasterize_regions(positive_regions, negative_regions, size)

    arr = np.array(mask, dtype=np.uint8)
    unique_values = np.unique(arr)
    if not set(unique_values.tolist()).issubset({0, 1}):
        raise ValueError(f'Unexpected values in {sample.xml_path}: {unique_values.tolist()}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode='L').save(output_path)

    if verbose:
        print(
            f'[OK] {sample.xml_path.name} -> {output_path} '
            f'size={size}, regions={len(positive_regions)}, values={unique_values.tolist()}'
        )

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MoNuSeg XML annotations to uint8 ground-truth masks with only 0 and 1.'
    )
    parser.add_argument(
        '--monuseg_root',
        type=str,
        default=DEFAULT_MONUSEG_ROOT,
        help=f'MoNuSeg root directory. Default: {DEFAULT_MONUSEG_ROOT}'
    )
    parser.add_argument(
        '--out_root',
        type=str,
        default=None,
        help='Output directory. Default: <monuseg_root>/ground_truth_01'
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help='Save all masks directly under out_root instead of split subdirectories.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output masks.'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only list discovered XML/image pairs without writing masks.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print details for every converted mask.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.monuseg_root)
    if not root.exists():
        raise FileNotFoundError(f'MoNuSeg root does not exist: {root}')

    out_root = Path(args.out_root) if args.out_root else root / 'ground_truth_01'
    samples = discover_samples(root)
    if not samples:
        raise ValueError(f'No XML annotation files found under: {root}')

    converted = 0
    skipped = 0
    for sample in samples:
        if args.flat:
            output_path = out_root / f'{sample.xml_path.stem}.png'
        else:
            output_path = out_root / sample.split / f'{sample.xml_path.stem}.png'

        if args.dry_run:
            print(f'[PAIR] split={sample.split} xml={sample.xml_path} image={sample.image_path} out={output_path}')
            continue

        changed = convert_one(sample, output_path, args.overwrite, args.verbose)
        if changed:
            converted += 1
        else:
            skipped += 1

    if args.dry_run:
        print(f'[DONE] Found {len(samples)} XML/image pairs.')
    else:
        print(f'[DONE] converted={converted}, skipped={skipped}, out_root={out_root}')


if __name__ == '__main__':
    main()
