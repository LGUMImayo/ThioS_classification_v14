#!/usr/bin/env python3
"""
ThioS Data Preparation Script - V2 (Improved Preprocessing)

This script converts CZI fluorescent images and Arivis-exported label masks
into a format suitable for training a multiclass segmentation model.

V2 Improvements:
- Intensity windowing instead of CLAHE (cleaner background, better contrast)
- Single-channel green (R=0, G=intensity, B=0) to match AF488 fluorescence
- Configurable intensity range (default: 300-2200 like Arivis visualization)
- Optional gamma correction

Tasks:
1. Convert single-channel AF488 CZI files to pseudo-RGB TIFF (green channel)
2. Merge multiple class label masks into a single multiclass mask
   - Priority: biomarker classes > background (for overlapping regions)
   - Classes: 0=unlabeled, 1=background, 2=diffuse, 3=plaque, 4=tangle

Usage:
    python prepare_thios_data_v2.py --input_czi_dir <path> --input_label_dir <path> --output_dir <path> \
        --intensity_min 300 --intensity_max 2200

Author: ThioS Classification Pipeline
Date: January 2026
"""

import os
import sys
import argparse
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from skimage import io, exposure, util
from skimage.util import img_as_ubyte
import tifffile
import pylibCZIrw.czi as pyczi  # Use pylibCZIrw for better compression support (zstd)
from tqdm import tqdm


# Class mapping (order matters for priority - higher index = higher priority)
CLASS_MAP = {
    'background': 1,
    'diffuse': 2,
    'plaque': 3,
    'tangle': 4,
    'tangles': 4,
}

CLASS_NAMES = {0: 'unlabeled', 1: 'background', 2: 'diffuse', 3: 'plaque', 4: 'tangle'}
NUM_CLASSES = 5


def find_label_files(label_dir: str, base_name: str) -> Dict[str, str]:
    """
    Find all label files for a given image base name.
    """
    found_labels = {}
    
    for class_name, class_id in CLASS_MAP.items():
        patterns = [
            f"{base_name}_{class_name}.ome.tiff",
            f"{base_name}_{class_name.capitalize()}.ome.tiff",
            f"{base_name}_{class_name.upper()}.ome.tiff",
            f"{base_name}_{class_name}s.ome.tiff",
            f"{base_name}_{class_name.capitalize()}s.ome.tiff",
        ]
        
        for pattern in patterns:
            label_path = os.path.join(label_dir, pattern)
            if os.path.exists(label_path):
                canonical_name = 'tangle' if class_name == 'tangles' else class_name
                found_labels[canonical_name] = label_path
                break
    
    return found_labels


# Maximum allowed size (in pixels) for each biomarker class
MAX_OBJECT_SIZE = {
    'background': None,
    'diffuse': 200,
    'plaque': 250,
    'tangle': 100,
}


def filter_outlier_rois(label_mask: np.ndarray, max_size: int, class_name: str) -> Tuple[np.ndarray, int]:
    """Remove ROIs larger than max_size from the label mask."""
    if max_size is None:
        return label_mask, 0
    
    unique_ids = np.unique(label_mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    removed_count = 0
    filtered_mask = label_mask.copy()
    
    for roi_id in unique_ids:
        roi_mask = label_mask == roi_id
        
        rows = np.any(roi_mask, axis=1)
        cols = np.any(roi_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            continue
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        max_dim = max(height, width)
        
        if max_dim > max_size:
            filtered_mask[roi_mask] = 0
            removed_count += 1
            print(f"    Removed {class_name} outlier ROI {roi_id}: {width}x{height} px")
    
    return filtered_mask, removed_count


def align_label_to_image(label_mask: np.ndarray, image_shape: Tuple[int, int], max_diff: int = 20) -> Optional[np.ndarray]:
    """
    Align label mask to image shape by padding or cropping.
    
    Handles small differences in dimensions (up to max_diff pixels).
    Returns None if the difference is too large.
    """
    label_h, label_w = label_mask.shape
    img_h, img_w = image_shape
    
    diff_h = abs(img_h - label_h)
    diff_w = abs(img_w - label_w)
    
    # If difference is too large, can't align
    if diff_h > max_diff or diff_w > max_diff:
        return None
    
    # Create output mask
    aligned_mask = np.zeros(image_shape, dtype=label_mask.dtype)
    
    # Calculate overlap region
    copy_h = min(label_h, img_h)
    copy_w = min(label_w, img_w)
    
    # Copy the overlapping region (top-left aligned)
    aligned_mask[:copy_h, :copy_w] = label_mask[:copy_h, :copy_w]
    
    return aligned_mask


def merge_labels_with_priority(label_files: Dict[str, str], image_shape: Tuple[int, int]) -> np.ndarray:
    """Merge multiple label masks into a single multiclass mask with priority handling."""
    merged_mask = np.zeros(image_shape, dtype=np.uint8)
    
    priority_order = ['background', 'diffuse', 'plaque', 'tangle']
    
    for class_name in priority_order:
        if class_name not in label_files:
            continue
            
        label_path = label_files[class_name]
        class_id = CLASS_MAP[class_name]
        
        try:
            label_mask = tifffile.imread(label_path)
            
            if label_mask.ndim > 2:
                label_mask = label_mask.squeeze()
            
            if label_mask.shape != image_shape:
                print(f"  Note: Label shape {label_mask.shape} != image shape {image_shape}, attempting alignment...")
                aligned_mask = align_label_to_image(label_mask, image_shape)
                if aligned_mask is None:
                    print(f"  Warning: Shape difference too large, skipping {class_name}")
                    continue
                label_mask = aligned_mask
                print(f"  Successfully aligned {class_name} label to image shape")
            
            max_size = MAX_OBJECT_SIZE.get(class_name)
            if max_size is not None:
                label_mask, removed = filter_outlier_rois(label_mask, max_size, class_name)
                if removed > 0:
                    print(f"  Filtered {removed} outlier(s) from {class_name}")
            
            class_pixels = label_mask > 0
            merged_mask[class_pixels] = class_id
            
        except Exception as e:
            print(f"  Warning: Failed to load {label_path}: {e}")
            continue
    
    return merged_mask


def convert_czi_with_intensity_window(
    czi_path: str,
    intensity_min: int = None,
    intensity_max: int = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    min_floor: int = 50,
    max_ceiling: int = 20000,
    gamma: float = 1.0,
    pure_green: bool = True
) -> np.ndarray:
    """
    Convert single-channel AF488 fluorescent CZI to pseudo-RGB image.
    
    Uses ADAPTIVE intensity windowing based on percentiles for each image,
    which handles varying exposure levels across different slides.
    
    Args:
        czi_path: Path to CZI file
        intensity_min: Fixed lower bound (if None, use percentile-based)
        intensity_max: Fixed upper bound (if None, use percentile-based)
        lower_percentile: Percentile for lower bound (default 1.0, clips darkest 1%)
        upper_percentile: Percentile for upper bound (default 99.5, clips brightest 0.5%)
        min_floor: Minimum value for lower bound (prevents too-dark images)
        max_ceiling: Maximum value for upper bound (prevents clipping real signal)
        gamma: Gamma correction (1.0 = no correction, <1 = brighter, >1 = darker)
        pure_green: If True, use only green channel (R=B=0). If False, add slight tint.
        
    Returns:
        3-channel RGB image (uint8), (original_min, original_max), (used_min, used_max)
    """
    with pyczi.open_czi(czi_path) as czi_doc:
        # Read channel 0 (AF488)
        data = czi_doc.read(plane={'C': 0})
        
    # Squeeze to 2D (pylibCZIrw returns HxWxC)
    data_2d = np.squeeze(data)
    
    # Get original data range for logging
    original_min = int(data_2d.min())
    original_max = int(data_2d.max())
    
    # Compute adaptive intensity window using percentiles
    if intensity_min is None:
        # Use percentile for lower bound, but ensure minimum floor
        p_low = np.percentile(data_2d, lower_percentile)
        intensity_min = max(int(p_low), min_floor)
    
    if intensity_max is None:
        # Use percentile for upper bound, but ensure maximum ceiling
        p_high = np.percentile(data_2d, upper_percentile)
        intensity_max = min(int(p_high), max_ceiling)
    
    # Ensure valid range
    if intensity_max <= intensity_min:
        intensity_max = intensity_min + 1
    
    used_min = intensity_min
    used_max = intensity_max
    
    # Apply intensity windowing (clip to [intensity_min, intensity_max])
    data_clipped = np.clip(data_2d.astype(np.float32), intensity_min, intensity_max)
    
    # Normalize to [0, 1]
    data_norm = (data_clipped - intensity_min) / (intensity_max - intensity_min)
    
    # Apply gamma correction if requested
    if gamma != 1.0:
        data_norm = np.power(data_norm, gamma)
    
    # Convert to 8-bit
    data_8bit = (data_norm * 255).astype(np.uint8)
    
    # Create pseudo-RGB
    rgb_image = np.zeros((*data_8bit.shape, 3), dtype=np.uint8)
    
    if pure_green:
        # Pure green channel (R=0, G=intensity, B=0)
        rgb_image[:, :, 1] = data_8bit
    else:
        # Add slight red/blue tint for potentially better CNN feature extraction
        rgb_image[:, :, 0] = (data_8bit * 0.1).astype(np.uint8)
        rgb_image[:, :, 1] = data_8bit
        rgb_image[:, :, 2] = (data_8bit * 0.1).astype(np.uint8)
    
    return rgb_image, (original_min, original_max), (used_min, used_max)


def process_dataset(
    input_czi_dir: str,
    input_label_dir: str,
    output_dir: str,
    intensity_min: int = None,
    intensity_max: int = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    min_floor: int = 50,
    max_ceiling: int = 20000,
    gamma: float = 1.0,
    pure_green: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Process all CZI files and corresponding labels into training format.
    
    Uses ADAPTIVE intensity windowing (percentile-based) by default.
    Set intensity_min/intensity_max to use fixed values instead.
    
    Output structure:
    output_dir/
        images/
            {base_name}.tiff  (pseudo-RGB)
        labels/
            {base_name}.png   (multiclass mask)
    """
    images_out = os.path.join(output_dir, 'images')
    labels_out = os.path.join(output_dir, 'labels')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    
    czi_files = glob.glob(os.path.join(input_czi_dir, '*.czi'))
    
    stats = {
        'total': len(czi_files),
        'processed': 0,
        'skipped_no_labels': 0,
        'skipped_error': 0,
        'class_pixels': {name: 0 for name in CLASS_NAMES.values()},
        'intensity_windows': []  # Track per-image windows
    }
    
    print(f"\nProcessing {len(czi_files)} CZI files...")
    print(f"Input CZI: {input_czi_dir}")
    print(f"Input Labels: {input_label_dir}")
    print(f"Output: {output_dir}")
    if intensity_min is not None and intensity_max is not None:
        print(f"Intensity window: FIXED [{intensity_min}, {intensity_max}]")
    else:
        print(f"Intensity window: ADAPTIVE (percentiles {lower_percentile}%-{upper_percentile}%)")
        print(f"  Floor/Ceiling bounds: [{min_floor}, {max_ceiling}]")
    print(f"Gamma: {gamma}")
    print(f"Pure green: {pure_green}")
    print("="*60)
    
    for czi_path in tqdm(czi_files, desc="Processing"):
        try:
            base_name = Path(czi_path).stem
            
            if verbose:
                print(f"\n[{base_name}]")
            
            # Find corresponding label files
            label_files = find_label_files(input_label_dir, base_name)
            
            if not label_files:
                if verbose:
                    print(f"  Skipping: No label files found")
                stats['skipped_no_labels'] += 1
                continue
            
            if verbose:
                print(f"  Found labels: {list(label_files.keys())}")
            
            # Convert CZI with intensity windowing (adaptive or fixed)
            rgb_image, (orig_min, orig_max), (used_min, used_max) = convert_czi_with_intensity_window(
                czi_path,
                intensity_min=intensity_min,
                intensity_max=intensity_max,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                min_floor=min_floor,
                max_ceiling=max_ceiling,
                gamma=gamma,
                pure_green=pure_green
            )
            image_shape = rgb_image.shape[:2]
            
            # Track intensity window used
            stats['intensity_windows'].append({
                'name': base_name,
                'original': (orig_min, orig_max),
                'used': (used_min, used_max)
            })
            
            if verbose:
                print(f"  Image shape: {rgb_image.shape}")
                print(f"  Original intensity range: [{orig_min}, {orig_max}]")
                print(f"  Used intensity window: [{used_min}, {used_max}]")
            
            # Merge labels with priority
            merged_mask = merge_labels_with_priority(label_files, image_shape)
            
            # Count class pixels
            unique, counts = np.unique(merged_mask, return_counts=True)
            for u, c in zip(unique, counts):
                class_name = CLASS_NAMES.get(u, 'unknown')
                stats['class_pixels'][class_name] = stats['class_pixels'].get(class_name, 0) + c
                if verbose:
                    print(f"  {class_name}: {c} pixels ({100*c/merged_mask.size:.2f}%)")
            
            # Save outputs
            image_path = os.path.join(images_out, f"{base_name}.tiff")
            label_path = os.path.join(labels_out, f"{base_name}.png")
            
            tifffile.imwrite(image_path, rgb_image, compression='LZW')
            io.imsave(label_path, merged_mask, check_contrast=False)
            
            stats['processed'] += 1
            
            if verbose:
                print(f"  Saved: {base_name}")
            
        except Exception as e:
            print(f"  Error processing {czi_path}: {e}")
            stats['skipped_error'] += 1
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total CZI files: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (no labels): {stats['skipped_no_labels']}")
    print(f"Skipped (errors): {stats['skipped_error']}")
    print("\nClass pixel distribution (all images):")
    total_pixels = sum(stats['class_pixels'].values())
    for class_name, count in stats['class_pixels'].items():
        pct = 100 * count / total_pixels if total_pixels > 0 else 0
        print(f"  {class_name}: {count:,} ({pct:.2f}%)")
    
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ThioS CZI images and labels for segmentation training (V2 with adaptive windowing)"
    )
    
    parser.add_argument(
        "--input_czi_dir", type=str, required=True,
        help="Directory containing CZI fluorescent images"
    )
    parser.add_argument(
        "--input_label_dir", type=str, required=True,
        help="Directory containing Arivis-exported label masks"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed data"
    )
    
    # Adaptive windowing (default)
    parser.add_argument(
        "--lower_percentile", type=float, default=1.0,
        help="Lower percentile for adaptive windowing (default: 1.0)"
    )
    parser.add_argument(
        "--upper_percentile", type=float, default=99.5,
        help="Upper percentile for adaptive windowing (default: 99.5)"
    )
    parser.add_argument(
        "--min_floor", type=int, default=50,
        help="Minimum floor for lower bound (default: 50)"
    )
    parser.add_argument(
        "--max_ceiling", type=int, default=20000,
        help="Maximum ceiling for upper bound (default: 20000)"
    )
    
    # Fixed windowing (optional, overrides adaptive)
    parser.add_argument(
        "--intensity_min", type=int, default=None,
        help="Fixed lower bound (overrides adaptive windowing)"
    )
    parser.add_argument(
        "--intensity_max", type=int, default=None,
        help="Fixed upper bound (overrides adaptive windowing)"
    )
    
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Gamma correction (1.0 = none, <1 = brighter, >1 = darker)"
    )
    parser.add_argument(
        "--pure_green", action="store_true", default=True,
        help="Use pure green channel (R=B=0)"
    )
    parser.add_argument(
        "--add_tint", action="store_true",
        help="Add slight R/B tint (10%) instead of pure green"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    pure_green = not args.add_tint
    
    stats = process_dataset(
        input_czi_dir=args.input_czi_dir,
        input_label_dir=args.input_label_dir,
        output_dir=args.output_dir,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        min_floor=args.min_floor,
        max_ceiling=args.max_ceiling,
        gamma=args.gamma,
        pure_green=pure_green,
        verbose=args.verbose
    )
    
    # Print intensity window summary
    if stats['intensity_windows']:
        print("\n" + "="*60)
        print("INTENSITY WINDOW SUMMARY (per image)")
        print("="*60)
        for info in stats['intensity_windows']:
            print(f"  {info['name']}: original [{info['original'][0]}, {info['original'][1]}] -> used [{info['used'][0]}, {info['used'][1]}]")
    
    print(f"\nDone! Processed {stats['processed']} images.")
