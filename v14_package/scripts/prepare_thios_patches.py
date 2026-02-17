#!/usr/bin/env python3
"""
ThioS Multiclass Patch Extraction Script

This script extracts training patches from full-size images and multiclass labels,
implementing smart sampling strategies to handle class imbalance with limited data.

Features:
1. Sliding window patch extraction
2. Stratified sampling to ensure all classes are represented
3. Overlap-based patch selection for rare classes
4. Class distribution balancing

Usage:
    python prepare_thios_patches.py --config_file <path_to_config.json>

Author: ThioS Classification Pipeline
Date: January 2026
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
from skimage import io, util
from tqdm import tqdm


# Class configuration
NUM_CLASSES = 5  # 0=unlabeled, 1=background, 2=diffuse, 3=plaque, 4=tangle
BIOMARKER_CLASSES = [2, 3, 4]  # diffuse, plaque, tangle
BACKGROUND_CLASS = 1
UNLABELED_CLASS = 0

CLASS_NAMES = {0: 'unlabeled', 1: 'background', 2: 'diffuse', 3: 'plaque', 4: 'tangle'}


def parse_config(config_path: str) -> Dict:
    """Load and parse configuration file."""
    with open(config_path) as f:
        config = json.load(f)
    
    # Ensure tuple types
    config['patch_size'] = tuple(config['patch_size'])
    config['step_size'] = tuple(config['step_size'])
    
    return config


def get_image_paths(directory: str) -> List[str]:
    """Get all image file paths from a directory."""
    extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    paths = []
    
    for ext in extensions:
        paths.extend(Path(directory).glob(f'*{ext}'))
        paths.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return sorted([str(p) for p in paths])


def compute_patch_class_distribution(label_patch: np.ndarray) -> Dict[int, float]:
    """
    Compute the class distribution within a patch.
    
    Returns:
        Dictionary mapping class ID to fraction of pixels
    """
    total_pixels = label_patch.size
    distribution = {}
    
    for class_id in range(NUM_CLASSES):
        count = np.sum(label_patch == class_id)
        distribution[class_id] = count / total_pixels
    
    return distribution


def classify_patch(distribution: Dict[int, float], config: Dict) -> Tuple[str, bool]:
    """
    Classify a patch based on its class distribution.
    
    Returns:
        Tuple of (patch_type, should_save)
        patch_type: 'biomarker', 'background', 'mixed', 'empty'
    """
    min_biomarker_ratio = config.get('min_biomarker_ratio', 0.01)  # 1%
    min_background_ratio = config.get('min_background_ratio', 0.05)  # 5%
    
    # Calculate total biomarker and background pixels
    biomarker_ratio = sum(distribution.get(c, 0) for c in BIOMARKER_CLASSES)
    background_ratio = distribution.get(BACKGROUND_CLASS, 0)
    unlabeled_ratio = distribution.get(UNLABELED_CLASS, 0)
    
    # Classification logic
    if biomarker_ratio >= min_biomarker_ratio:
        return 'biomarker', True
    elif background_ratio >= min_background_ratio:
        return 'background', True
    elif unlabeled_ratio < 0.95:  # At least 5% labeled
        return 'mixed', True
    else:
        return 'empty', False


def extract_patches(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int],
    step_size: Tuple[int, int],
    config: Dict
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """
    Extract patches from image and label pair using sliding window.
    
    Returns:
        Tuple of (image_patches, label_patches, patch_stats)
    """
    patch_h, patch_w = patch_size
    step_h, step_w = step_size
    img_h, img_w = image.shape[:2]
    
    image_patches = []
    label_patches = []
    
    stats = {
        'total_windows': 0,
        'saved_patches': 0,
        'biomarker': 0,
        'background': 0,
        'mixed': 0,
        'empty': 0,
    }
    
    # Sample tracking for class balancing
    background_samples = []
    
    # Sliding window
    for y in range(0, img_h - patch_h + 1, step_h):
        for x in range(0, img_w - patch_w + 1, step_w):
            stats['total_windows'] += 1
            
            # Extract patches
            if image.ndim == 3:
                img_patch = image[y:y+patch_h, x:x+patch_w, :]
            else:
                img_patch = image[y:y+patch_h, x:x+patch_w]
            
            lbl_patch = label[y:y+patch_h, x:x+patch_w]
            
            # Analyze patch
            distribution = compute_patch_class_distribution(lbl_patch)
            patch_type, should_save = classify_patch(distribution, config)
            stats[patch_type] += 1
            
            if should_save:
                if patch_type == 'biomarker':
                    # Always save biomarker patches (they're precious)
                    image_patches.append(img_patch)
                    label_patches.append(lbl_patch)
                    stats['saved_patches'] += 1
                elif patch_type == 'background':
                    # Store background patches for downsampling
                    background_samples.append((img_patch, lbl_patch))
                elif patch_type == 'mixed':
                    # Save mixed patches with some probability
                    if random.random() < config.get('mixed_keep_ratio', 0.5):
                        image_patches.append(img_patch)
                        label_patches.append(lbl_patch)
                        stats['saved_patches'] += 1
    
    # Downsample background patches to balance classes
    # Keep ratio relative to biomarker patches
    max_background = max(
        stats['biomarker'] * config.get('background_ratio', 2),
        config.get('min_background_patches', 10)
    )
    
    if len(background_samples) > max_background:
        background_samples = random.sample(background_samples, int(max_background))
    
    for img_patch, lbl_patch in background_samples:
        image_patches.append(img_patch)
        label_patches.append(lbl_patch)
        stats['saved_patches'] += 1
    
    return image_patches, label_patches, stats


def prepare_patches(config: Dict) -> None:
    """
    Main function to prepare patches from full images and labels.
    """
    # Directories
    image_dir = config['images_path']
    label_dir = config['labels_path']
    output_dir = config['save_path']
    
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get file lists
    image_files = get_image_paths(image_dir)
    label_files = get_image_paths(label_dir)
    
    print(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    # Build filename mappings
    image_dict = {Path(f).stem: f for f in image_files}
    label_dict = {Path(f).stem: f for f in label_files}
    
    # Find matching pairs
    common_names = set(image_dict.keys()) & set(label_dict.keys())
    print(f"Found {len(common_names)} matching image-label pairs")
    
    if not common_names:
        print("Error: No matching image-label pairs found!")
        print(f"Image stems: {list(image_dict.keys())[:5]}...")
        print(f"Label stems: {list(label_dict.keys())[:5]}...")
        sys.exit(1)
    
    # Processing parameters
    patch_size = config['patch_size']
    step_size = config['step_size']
    
    # Global statistics
    total_stats = {
        'images_processed': 0,
        'total_patches': 0,
        'class_pixels': {i: 0 for i in range(NUM_CLASSES)},
    }
    
    patch_idx = 0
    
    for name in tqdm(sorted(common_names), desc="Processing images"):
        image_path = image_dict[name]
        label_path = label_dict[name]
        
        # Load image and label
        image = io.imread(image_path)
        label = io.imread(label_path)
        
        # Validate dimensions
        if image.shape[:2] != label.shape[:2]:
            print(f"Warning: Shape mismatch for {name}")
            print(f"  Image: {image.shape}, Label: {label.shape}")
            continue
        
        # Skip if too small
        if image.shape[0] < patch_size[0] or image.shape[1] < patch_size[1]:
            print(f"Warning: {name} too small ({image.shape[:2]}) for patch size {patch_size}")
            continue
        
        # Extract patches
        img_patches, lbl_patches, stats = extract_patches(
            image, label, patch_size, step_size, config
        )
        
        # Save patches
        for img_patch, lbl_patch in zip(img_patches, lbl_patches):
            # Ensure correct types
            if img_patch.dtype != np.uint8:
                if img_patch.max() <= 1.0:
                    img_patch = (img_patch * 255).astype(np.uint8)
                else:
                    img_patch = img_patch.astype(np.uint8)
            
            lbl_patch = lbl_patch.astype(np.uint8)
            
            # Save files
            img_fname = f"patch_{patch_idx:06d}_img_{name}.tif"
            lbl_fname = f"patch_{patch_idx:06d}_mask_{name}.png"
            
            io.imsave(
                os.path.join(output_image_dir, img_fname),
                img_patch,
                check_contrast=False
            )
            io.imsave(
                os.path.join(output_label_dir, lbl_fname),
                lbl_patch,
                check_contrast=False
            )
            
            # Update class statistics
            for class_id in range(NUM_CLASSES):
                total_stats['class_pixels'][class_id] += np.sum(lbl_patch == class_id)
            
            patch_idx += 1
        
        total_stats['images_processed'] += 1
        total_stats['total_patches'] += len(img_patches)
    
    # Print summary
    print("\n" + "="*60)
    print("PATCH EXTRACTION SUMMARY")
    print("="*60)
    print(f"Images processed: {total_stats['images_processed']}")
    print(f"Total patches saved: {total_stats['total_patches']}")
    print(f"Patches saved to: {output_dir}")
    print("\nClass pixel distribution:")
    total_pixels = sum(total_stats['class_pixels'].values())
    for class_id, count in total_stats['class_pixels'].items():
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {CLASS_NAMES[class_id]}: {count:,} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training patches for ThioS multiclass segmentation"
    )
    parser.add_argument(
        "--config_file", type=str, required=True,
        help="Path to JSON configuration file"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)
    
    config = parse_config(args.config_file)
    prepare_patches(config)


if __name__ == "__main__":
    main()
