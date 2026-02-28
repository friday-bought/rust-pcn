#!/usr/bin/env python3
"""Convert various image datasets to CIFAR-10-like binary format.

Output format per record: [label: u8] [R: H*W * u8] [G: H*W * u8] [B: H*W * u8]
All images resized to 32x32 using PIL bilinear interpolation.

Usage:
    python3 convert-datasets.py --dataset cifar100 --input /datasets/cifar-100/cifar-100-binary --output /datasets/cifar100-converted
    python3 convert-datasets.py --dataset stl10 --input /datasets/stl-10/stl10_binary --output /datasets/stl10-converted
    python3 convert-datasets.py --dataset flowers102 --input /datasets/flowers-102 --output /datasets/flowers102-converted
    python3 convert-datasets.py --dataset tiny-imagenet --input /datasets/tiny-imagenet-200/tiny-imagenet-200 --output /datasets/tinyimagenet-converted
"""

import argparse
import os
import struct
import sys
import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system(f"{sys.executable} -m pip install Pillow")
    from PIL import Image

TARGET_SIZE = 32
RECORD_SIZE = 1 + 3 * TARGET_SIZE * TARGET_SIZE  # 3073 bytes


def image_to_cifar_record(img_array, label):
    """Convert an HWC uint8 image (32x32x3) to CIFAR binary record."""
    assert img_array.shape == (32, 32, 3), f"Expected (32,32,3), got {img_array.shape}"
    r = img_array[:, :, 0].flatten()
    g = img_array[:, :, 1].flatten()
    b = img_array[:, :, 2].flatten()
    return struct.pack('B', label) + r.tobytes() + g.tobytes() + b.tobytes()


def resize_image(img, size=TARGET_SIZE):
    """Resize PIL image to size x size using bilinear interpolation."""
    return img.resize((size, size), Image.BILINEAR).convert('RGB')


def write_batch(records, path, batch_num):
    """Write a batch of records to a binary file."""
    batch_path = path / f"data_batch_{batch_num}.bin"
    with open(batch_path, 'wb') as f:
        for record in records:
            f.write(record)
    print(f"  Wrote {len(records)} records to {batch_path}")


def write_test_batch(records, path):
    """Write test records to test_batch.bin."""
    batch_path = path / "test_batch.bin"
    with open(batch_path, 'wb') as f:
        for record in records:
            f.write(record)
    print(f"  Wrote {len(records)} test records to {batch_path}")


def split_into_batches(records, batch_size=10000):
    """Split records into batches of batch_size."""
    batches = []
    for i in range(0, len(records), batch_size):
        batches.append(records[i:i+batch_size])
    return batches


# ─── CIFAR-100 ──────────────────────────────────────────────────────────────

def convert_cifar100(input_dir, output_dir):
    """Convert CIFAR-100 binary to CIFAR-10-like format.
    
    CIFAR-100 record: [coarse_label: u8][fine_label: u8][pixels: 3072 * u8]
    We use fine_label (100 classes) and output in CIFAR-10 format.
    """
    print("Converting CIFAR-100...")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_path = input_dir / "train.bin"
    data = open(train_path, 'rb').read()
    cifar100_record = 2 + 3072  # 2 label bytes + pixels
    num_train = len(data) // cifar100_record
    print(f"  Training samples: {num_train}")
    
    train_records = []
    for i in range(num_train):
        offset = i * cifar100_record
        fine_label = data[offset + 1]  # skip coarse_label, use fine_label
        pixels = data[offset + 2: offset + cifar100_record]
        record = struct.pack('B', fine_label) + pixels
        train_records.append(record)
    
    # Split into batches of 10000
    batches = split_into_batches(train_records)
    for idx, batch in enumerate(batches):
        write_batch(batch, output_dir, idx + 1)

    # Test
    test_path = input_dir / "test.bin"
    data = open(test_path, 'rb').read()
    num_test = len(data) // cifar100_record
    print(f"  Test samples: {num_test}")
    
    test_records = []
    for i in range(num_test):
        offset = i * cifar100_record
        fine_label = data[offset + 1]
        pixels = data[offset + 2: offset + cifar100_record]
        record = struct.pack('B', fine_label) + pixels
        test_records.append(record)
    
    write_test_batch(test_records, output_dir)
    
    # Write metadata
    with open(output_dir / "metadata.txt", 'w') as f:
        f.write(f"dataset=cifar100\nnum_classes=100\ntrain_samples={num_train}\ntest_samples={num_test}\n")
    print(f"  Done! Output at {output_dir}")


# ─── STL-10 ─────────────────────────────────────────────────────────────────

def convert_stl10(input_dir, output_dir):
    """Convert STL-10 binary (96x96) to CIFAR-10-like format (32x32).
    
    STL-10 format:
    - train_X.bin: raw uint8 pixels, 96×96×3, column-major per channel
    - train_y.bin: labels (1-indexed)
    """
    print("Converting STL-10 (96×96 → 32×32)...")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_size = 96
    stl_pixels = stl_size * stl_size * 3

    for split, x_file, y_file in [
        ("train", "train_X.bin", "train_y.bin"),
        ("test", "test_X.bin", "test_y.bin"),
    ]:
        x_path = input_dir / x_file
        y_path = input_dir / y_file
        
        x_data = np.fromfile(str(x_path), dtype=np.uint8)
        y_data = np.fromfile(str(y_path), dtype=np.uint8)
        
        num_images = len(x_data) // stl_pixels
        print(f"  {split}: {num_images} images")
        
        x_data = x_data.reshape(num_images, 3, stl_size, stl_size)
        # STL-10 is CHW format, transpose to HWC
        x_data = x_data.transpose(0, 2, 3, 1)
        # Labels are 1-indexed, convert to 0-indexed
        y_data = y_data - 1
        
        records = []
        for i in range(num_images):
            img = Image.fromarray(x_data[i])
            img = resize_image(img)
            img_array = np.array(img)
            records.append(image_to_cifar_record(img_array, int(y_data[i])))
            
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i+1}/{num_images}")
        
        if split == "train":
            batches = split_into_batches(records)
            for idx, batch in enumerate(batches):
                write_batch(batch, output_dir, idx + 1)
        else:
            write_test_batch(records, output_dir)
    
    with open(output_dir / "metadata.txt", 'w') as f:
        f.write(f"dataset=stl10\nnum_classes=10\n")
    print(f"  Done! Output at {output_dir}")


# ─── Flowers-102 ────────────────────────────────────────────────────────────

def convert_flowers102(input_dir, output_dir):
    """Convert Oxford Flowers-102 to CIFAR-10-like format.
    
    Uses imagelabels.mat and setid.mat for labels and splits.
    """
    print("Converting Flowers-102...")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse .mat files (simple MATLAB v5 format)
    try:
        import scipy.io as sio
    except ImportError:
        print("Installing scipy for .mat parsing...")
        os.system(f"{sys.executable} -m pip install scipy")
        import scipy.io as sio
    
    labels_mat = sio.loadmat(str(input_dir / "imagelabels.mat"))
    setid_mat = sio.loadmat(str(input_dir / "setid.mat"))
    
    labels = labels_mat['labels'].flatten() - 1  # 1-indexed to 0-indexed
    train_ids = setid_mat['trnid'].flatten()
    val_ids = setid_mat['valid'].flatten()
    test_ids = setid_mat['tstid'].flatten()
    
    # Combine train + val for training
    train_ids = np.concatenate([train_ids, val_ids])
    
    jpg_dir = input_dir / "jpg"
    
    def load_images(ids, split_name):
        records = []
        for i, img_id in enumerate(ids):
            img_path = jpg_dir / f"image_{img_id:05d}.jpg"
            if not img_path.exists():
                print(f"    WARNING: {img_path} not found, skipping")
                continue
            img = Image.open(img_path)
            img = resize_image(img)
            img_array = np.array(img)
            label = int(labels[img_id - 1])  # img_id is 1-indexed
            records.append(image_to_cifar_record(img_array, label))
            
            if (i + 1) % 500 == 0:
                print(f"    {split_name}: {i+1}/{len(ids)}")
        return records
    
    print(f"  Train+Val IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")
    
    train_records = load_images(train_ids, "train")
    batches = split_into_batches(train_records)
    for idx, batch in enumerate(batches):
        write_batch(batch, output_dir, idx + 1)
    
    test_records = load_images(test_ids, "test")
    write_test_batch(test_records, output_dir)
    
    with open(output_dir / "metadata.txt", 'w') as f:
        f.write(f"dataset=flowers102\nnum_classes=102\ntrain_samples={len(train_records)}\ntest_samples={len(test_records)}\n")
    print(f"  Done! Output at {output_dir}")


# ─── Tiny ImageNet ──────────────────────────────────────────────────────────

def convert_tiny_imagenet(input_dir, output_dir):
    """Convert Tiny ImageNet-200 to CIFAR-10-like format (64x64 → 32x32)."""
    print("Converting Tiny ImageNet-200 (64×64 → 32×32)...")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build class-to-index mapping
    wnids_path = input_dir / "wnids.txt"
    with open(wnids_path) as f:
        wnids = [line.strip() for line in f if line.strip()]
    class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
    
    # Train
    train_dir = input_dir / "train"
    train_records = []
    for class_name in sorted(os.listdir(train_dir)):
        class_dir = train_dir / class_name / "images"
        if not class_dir.is_dir():
            continue
        label = class_to_idx.get(class_name)
        if label is None:
            continue
        for img_name in sorted(os.listdir(class_dir)):
            if not img_name.endswith('.JPEG'):
                continue
            img = Image.open(class_dir / img_name)
            img = resize_image(img)
            img_array = np.array(img)
            if img_array.shape != (32, 32, 3):
                continue  # skip grayscale
            train_records.append(image_to_cifar_record(img_array, label))
        
        if (label + 1) % 20 == 0:
            print(f"    Train: processed {label+1}/200 classes ({len(train_records)} images)")
    
    print(f"  Total train images: {len(train_records)}")
    batches = split_into_batches(train_records)
    for idx, batch in enumerate(batches):
        write_batch(batch, output_dir, idx + 1)
    
    # Val (has annotations)
    val_dir = input_dir / "val"
    val_annotations = val_dir / "val_annotations.txt"
    test_records = []
    
    if val_annotations.exists():
        with open(val_annotations) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_name, class_name = parts[0], parts[1]
                label = class_to_idx.get(class_name)
                if label is None:
                    continue
                img_path = val_dir / "images" / img_name
                if not img_path.exists():
                    continue
                img = Image.open(img_path)
                img = resize_image(img)
                img_array = np.array(img)
                if img_array.shape != (32, 32, 3):
                    continue
                test_records.append(image_to_cifar_record(img_array, label))
    
    print(f"  Total test images: {len(test_records)}")
    write_test_batch(test_records, output_dir)
    
    with open(output_dir / "metadata.txt", 'w') as f:
        f.write(f"dataset=tiny-imagenet\nnum_classes=200\ntrain_samples={len(train_records)}\ntest_samples={len(test_records)}\n")
    print(f"  Done! Output at {output_dir}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert image datasets to CIFAR-10-like binary format")
    parser.add_argument('--dataset', required=True, choices=['cifar100', 'stl10', 'flowers102', 'tiny-imagenet', 'all'])
    parser.add_argument('--input', required=False, help="Input directory")
    parser.add_argument('--output', required=False, help="Output directory")
    parser.add_argument('--base-input', default="/datasets", help="Base input directory")
    parser.add_argument('--base-output', default="/datasets", help="Base output directory")
    args = parser.parse_args()

    converters = {
        'cifar100': (
            lambda: convert_cifar100(
                args.input or f"{args.base_input}/cifar-100/cifar-100-binary",
                args.output or f"{args.base_output}/cifar100-converted"
            )
        ),
        'stl10': (
            lambda: convert_stl10(
                args.input or f"{args.base_input}/stl-10/stl10_binary",
                args.output or f"{args.base_output}/stl10-converted"
            )
        ),
        'flowers102': (
            lambda: convert_flowers102(
                args.input or f"{args.base_input}/flowers-102",
                args.output or f"{args.base_output}/flowers102-converted"
            )
        ),
        'tiny-imagenet': (
            lambda: convert_tiny_imagenet(
                args.input or f"{args.base_input}/tiny-imagenet-200/tiny-imagenet-200",
                args.output or f"{args.base_output}/tinyimagenet-converted"
            )
        ),
    }

    if args.dataset == 'all':
        for name, converter in converters.items():
            print(f"\n{'='*60}")
            converter()
    else:
        converters[args.dataset]()


if __name__ == '__main__':
    main()
