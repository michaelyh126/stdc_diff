import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def to_binary_mask(mask, threshold=0):
    arr = np.array(mask)

    if arr.ndim == 3:
        arr = arr.max(axis=2)

    return (arr > threshold).astype(np.uint8)


def collect_images(input_path, recursive=False):
    if input_path.is_file():
        return [input_path]

    pattern = '**/*' if recursive else '*'
    return sorted(
        p for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def get_output_path(input_file, input_root, output_root, inplace):
    if inplace:
        return input_file

    if input_root.is_file():
        if output_root.suffix.lower() in IMAGE_SUFFIXES:
            return output_root
        return output_root / input_file.name

    return output_root / input_file.relative_to(input_root)


def convert_one(input_file, output_file, threshold=0):
    mask = Image.open(input_file)
    binary = to_binary_mask(mask, threshold=threshold)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode='L').save(output_file)

    return np.unique(binary)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert mask images to single-channel uint8 labels with only 0 and 1.'
    )
    parser.add_argument(
        'input',
        help='Input mask file or directory.'
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Output file or directory. Required unless --inplace is used.'
    )
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Overwrite input files directly.'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process directories recursively.'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=0,
        help='Pixels greater than this value become 1. Default: 0.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f'Input does not exist: {input_path}')

    if not args.inplace and args.output is None:
        raise ValueError('Output is required unless --inplace is used.')

    output_root = Path(args.output) if args.output else None
    image_files = collect_images(input_path, recursive=args.recursive)

    if len(image_files) == 0:
        raise ValueError(f'No image files found in: {input_path}')

    for input_file in image_files:
        output_file = get_output_path(
            input_file, input_path, output_root, args.inplace)
        values = convert_one(input_file, output_file, threshold=args.threshold)
        print(f'{input_file} -> {output_file}, values={values.tolist()}')

    print(f'Done. Processed {len(image_files)} file(s).')


if __name__ == '__main__':
    main()
