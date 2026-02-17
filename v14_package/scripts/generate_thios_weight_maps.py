#!/usr/bin/env python3
"""
ThioS Multiclass Weight Map Generator

Generates spatial attention weight maps for multiclass segmentation training.
Weight maps emphasize:
1. Biomarker class pixels (higher weight)
2. Class boundaries (highest weight for edge precision)
3. Rare classes get extra weight to address imbalance

Usage:
    python generate_thios_weight_maps.py --data_dir <path> --class_weights <json>

"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import ndimage
from skimage import io
from tqdm import tqdm


# Class configuration
NUM_CLASSES = 5
CLASS_NAMES = {0: 'unlabeled', 1: 'background', 2: 'diffuse', 3: 'plaque', 4: 'tangle'}

# Default class weights (can be overridden via config)
DEFAULT_CLASS_WEIGHTS = {
    0: 0.0,   # unlabeled - no weight (ignore in loss)
    1: 1.0,   # background - base weight
    2: 4.0,   # diffuse - higher weight (rarer class)
    3: 5.0,   # plaque - even higher (typically rarer)
    4: 6.0,   # tangle - highest (often rarest)
}

# Boundary detection settings
BOUNDARY_DILATION = 2  # pixels
BOUNDARY_WEIGHT_BOOST = 2.0  # additional weight for boundaries


def compute_class_weights_from_data(label_dir: str) -> Dict[int, float]:
    """
    Compute class weights inversely proportional to class frequency.
    
    Uses median frequency balancing: weight = median_freq / class_freq
    """
    label_files = glob.glob(os.path.join(label_dir, '*.png'))
    
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    total_pixels = 0
    
    print("Computing class frequencies...")
    for label_path in tqdm(label_files, desc="Scanning labels"):
        label = io.imread(label_path)
        for class_id in range(NUM_CLASSES):
            class_counts[class_id] += np.sum(label == class_id)
        total_pixels += label.size
    
    # Compute frequencies (avoid division by zero)
    frequencies = {i: max(count, 1) / total_pixels for i, count in class_counts.items()}
    
    # Median frequency balancing
    freq_values = [f for i, f in frequencies.items() if i > 0]  # Exclude unlabeled
    median_freq = np.median(freq_values) if freq_values else 1.0
    
    weights = {}
    for class_id, freq in frequencies.items():
        if class_id == 0:
            weights[class_id] = 0.0  # Unlabeled always 0
        else:
            # Weight inversely proportional to frequency, capped at 10x
            weights[class_id] = min(median_freq / freq, 10.0)
    
    print("\nComputed class weights:")
    for class_id, weight in weights.items():
        count = class_counts[class_id]
        print(f"  {CLASS_NAMES[class_id]}: {weight:.3f} (pixels: {count:,})")
    
    return weights


def detect_boundaries(label: np.ndarray, dilation_iterations: int = 2) -> np.ndarray:
    """
    Detect boundaries between different classes.
    
    Returns:
        Binary mask where True indicates boundary pixels
    """
    # For each class, find the boundary with other classes
    boundaries = np.zeros_like(label, dtype=bool)
    
    for class_id in range(NUM_CLASSES):
        if class_id == 0:
            continue  # Skip unlabeled
        
        class_mask = label == class_id
        
        if not np.any(class_mask):
            continue
        
        # Dilate the mask
        dilated = ndimage.binary_dilation(
            class_mask, 
            iterations=dilation_iterations
        )
        
        # Boundary is dilated minus original
        boundary = dilated & ~class_mask
        
        # Only keep boundaries that are adjacent to labeled regions
        boundaries |= boundary
    
    return boundaries


def create_weight_map(
    label: np.ndarray,
    class_weights: Dict[int, float],
    boundary_weight_boost: float = BOUNDARY_WEIGHT_BOOST,
    boundary_dilation: int = BOUNDARY_DILATION
) -> np.ndarray:
    """
    Create a weight map for the given label.
    
    Weight strategy:
    1. Base weight from class_weights dictionary
    2. Additional boost at class boundaries
    3. Maximum clipping to prevent extreme values
    """
    # Initialize with base weights
    weight_map = np.zeros_like(label, dtype=np.float32)
    
    for class_id, weight in class_weights.items():
        class_mask = label == class_id
        weight_map[class_mask] = weight
    
    # Detect and boost boundaries
    boundaries = detect_boundaries(label, boundary_dilation)
    weight_map[boundaries] += boundary_weight_boost
    
    # Clip to reasonable range
    weight_map = np.clip(weight_map, 0.0, 15.0)
    
    return weight_map


def generate_weight_maps(
    data_dir: str,
    class_weights: Optional[Dict[int, float]] = None,
    auto_compute_weights: bool = True
) -> None:
    """
    Generate weight maps for all patches in data_dir.
    
    Expected structure:
    data_dir/
        images/
        labels/
        weights/  (will be created)
    """
    label_dir = os.path.join(data_dir, 'labels')
    weight_dir = os.path.join(data_dir, 'weights')
    
    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found: {label_dir}")
        sys.exit(1)
    
    os.makedirs(weight_dir, exist_ok=True)
    
    # Determine class weights
    if class_weights is None:
        if auto_compute_weights:
            class_weights = compute_class_weights_from_data(label_dir)
        else:
            class_weights = DEFAULT_CLASS_WEIGHTS
            print("Using default class weights:")
            for class_id, weight in class_weights.items():
                print(f"  {CLASS_NAMES[class_id]}: {weight:.3f}")
    
    # Get label files
    label_files = glob.glob(os.path.join(label_dir, '*.png'))
    
    print(f"\nGenerating weight maps for {len(label_files)} patches...")
    
    for label_path in tqdm(label_files, desc="Generating weights"):
        # Load label
        label = io.imread(label_path)
        
        # Create weight map
        weight_map = create_weight_map(
            label, 
            class_weights,
            boundary_weight_boost=BOUNDARY_WEIGHT_BOOST,
            boundary_dilation=BOUNDARY_DILATION
        )
        
        # Construct output filename
        # Input: patch_XXX_mask_YYY.png
        # Output: patch_XXX_img_YYY.tif (to match image naming convention)
        fname = Path(label_path).stem
        # Replace _mask_ with _img_ to match image naming
        weight_fname = fname.replace('_mask_', '_img_') + '.tif'
        
        # Save weight map
        output_path = os.path.join(weight_dir, weight_fname)
        io.imsave(output_path, weight_map.astype(np.float32), check_contrast=False)
    
    # Save class weights for reference
    weights_config = os.path.join(data_dir, 'class_weights.json')
    with open(weights_config, 'w') as f:
        json.dump({str(k): v for k, v in class_weights.items()}, f, indent=2)
    
    print(f"\nWeight maps saved to: {weight_dir}")
    print(f"Class weights saved to: {weights_config}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate weight maps for ThioS multiclass segmentation"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing 'images' and 'labels' subfolders"
    )
    parser.add_argument(
        "--class_weights", type=str, default=None,
        help="JSON file with custom class weights {class_id: weight}"
    )
    parser.add_argument(
        "--no_auto_weights", action="store_true",
        help="Don't auto-compute weights from data; use defaults"
    )
    
    args = parser.parse_args()
    
    # Load custom weights if provided
    custom_weights = None
    if args.class_weights:
        if os.path.exists(args.class_weights):
            with open(args.class_weights) as f:
                custom_weights = {int(k): v for k, v in json.load(f).items()}
            print(f"Loaded custom class weights from: {args.class_weights}")
        else:
            print(f"Warning: Custom weights file not found: {args.class_weights}")
    
    generate_weight_maps(
        args.data_dir,
        class_weights=custom_weights,
        auto_compute_weights=not args.no_auto_weights
    )


if __name__ == "__main__":
    main()
