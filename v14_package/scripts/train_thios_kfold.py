#!/usr/bin/env python3
"""
ThioS Multiclass Segmentation Training Script with K-Fold Cross-Validation

Trains an EfficientNet-B4 + U-Net model for 5-class ThioS segmentation
using stratified k-fold cross-validation to ensure rare classes are 
properly represented in all folds.

Classes:
- 0: Unlabeled (ignored in loss)
- 1: Background
- 2: Diffuse amyloid plaque
- 3: Cored amyloid plaque
- 4: Tau tangle

Usage:
    python train_thios_kfold.py --config_file <path> --n_folds 5 --gpus 1

Author: ThioS Classification Pipeline
Date: January 2026
"""

import os
import sys
import json
import csv
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
# PyTorch 2.6+ defaults weights_only=True which breaks MONAI checkpoint loading
# (monai.utils.enums.TraceKeys is not in the safe globals list)
torch.serialization.add_safe_globals([])  # Initialize
import monai.utils.enums
torch.serialization.add_safe_globals([monai.utils.enums.TraceKeys])
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from tqdm import tqdm
from skimage import io
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from monai.data import list_data_collate
from monai.utils import set_determinism

# Import local modules
from unet2D_thios import ThioSUnet2D, ThioSDataset, PredDataset2D

# Class configuration
NUM_CLASSES = 5
CLASS_NAMES = {0: 'unlabeled', 1: 'background', 2: 'diffuse', 3: 'plaque', 4: 'tangle'}


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="ThioS K-Fold Cross-Validation Training")
    parser.add_argument(
        "--config_file", type=str, required=True,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--specific_fold", type=int, default=None,
        help="Train only a specific fold (0 to n_folds-1). If None, trains all folds."
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--strategy", type=str, default="auto",
        help="Training strategy (ddp, dp, auto)"
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu",
        help="Accelerator type (gpu, cpu)"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and parse configuration file."""
    with open(config_path) as f:
        config = json.load(f)
    
    # Merge model and dataset params
    args = {}
    args.update(config.get('dataset_params', {}))
    args.update(config.get('model_params', {}))
    
    # Ensure tuple types
    if 'patch_size' in args:
        args['patch_size'] = tuple(args['patch_size'])
    if 'pred_patch_size' in args:
        args['pred_patch_size'] = tuple(args['pred_patch_size'])
    
    return args


def get_image_paths(directory: str) -> list:
    """Get all image file paths from directory."""
    extensions = ['.tiff', '.tif', '.png']
    paths = []
    for ext in extensions:
        paths.extend(Path(directory).glob(f'*{ext}'))
        paths.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted([str(p) for p in paths])


def create_stratification_labels(pairs: list, num_classes: int) -> np.ndarray:
    """
    Create stratification labels for patches based on their dominant rare class.
    
    Strategy:
    - Assign each patch to its rarest class present (tangle > diffuse > plaque > background)
    - This ensures rare classes are distributed across all folds
    
    Args:
        pairs: List of (image_path, label_path) tuples
        num_classes: Number of classes (5)
        
    Returns:
        Array of stratification labels (one per patch)
    """
    print("\nCreating stratification labels for k-fold splitting...")
    
    strat_labels = []
    class_priority = [4, 2, 3, 1]  # tangle > diffuse > plaque > background
    
    for img_path, lbl_path in tqdm(pairs, desc="Analyzing patches"):
        label = io.imread(lbl_path)
        
        # Assign to rarest class present (based on priority)
        assigned = 1  # Default to background
        for class_id in class_priority:
            if np.any(label == class_id):
                assigned = class_id
                break
        
        strat_labels.append(assigned)
    
    strat_labels = np.array(strat_labels)
    
    # Report distribution
    print("\nStratification label distribution:")
    for c in range(num_classes):
        count = np.sum(strat_labels == c)
        print(f"  {CLASS_NAMES[c]} (class {c}): {count} patches ({100*count/len(strat_labels):.2f}%)")
    
    return strat_labels


def prepare_kfold_splits(args: dict, n_folds: int):
    """
    Prepare k-fold cross-validation splits with stratification.
    
    Returns:
        List of (train_ds, val_ds) tuples, one for each fold
    """
    image_dir = args['image_dir']
    label_dir = args['label_dir']
    
    print(f"\nLoading data from:")
    print(f"  Images: {image_dir}")
    print(f"  Labels: {label_dir}")
    
    # Get file paths
    image_paths = get_image_paths(image_dir)
    label_paths = get_image_paths(label_dir)
    
    print(f"Found {len(image_paths)} images and {len(label_paths)} labels")
    
    # Match pairs (assumes 1:1 sorted correspondence)
    if len(image_paths) != len(label_paths):
        print(f"ERROR: Mismatch in image/label counts!")
        sys.exit(1)
    
    pairs = list(zip(sorted(image_paths), sorted(label_paths)))
    print(f"Total pairs: {len(pairs)}")
    
    # Create stratification labels
    strat_labels = create_stratification_labels(pairs, NUM_CLASSES)
    
    # Stratified K-Fold split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print(f"\nCreating {n_folds}-fold stratified splits...")
    fold_splits = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pairs, strat_labels)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}:")
        print(f"    Train: {len(train_idx)} samples")
        print(f"    Val:   {len(val_idx)} samples")
        
        # Check class distribution in validation set
        val_classes = strat_labels[val_idx]
        print(f"    Val class distribution:")
        for c in range(1, NUM_CLASSES):  # Skip unlabeled
            count = np.sum(val_classes == c)
            print(f"      {CLASS_NAMES[c]}: {count} ({100*count/len(val_idx):.2f}%)")
        
        # Create file lists for this fold
        train_images = [pairs[i][0] for i in train_idx]
        train_labels = [pairs[i][1] for i in train_idx]
        val_images = [pairs[i][0] for i in val_idx]
        val_labels = [pairs[i][1] for i in val_idx]
        
        fold_splits.append({
            'train_images': train_images,
            'train_labels': train_labels,
            'val_images': val_images,
            'val_labels': val_labels,
            'train_indices': train_idx,
            'val_indices': val_idx,
        })
    
    return fold_splits


def train_single_fold(fold_idx: int, fold_data: dict, args: dict, cmd_args, base_log_dir: str):
    """Train a single fold."""
    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_idx + 1}")
    print("="*70)
    
    # Create fold-specific log directory
    fold_log_dir = os.path.join(base_log_dir, f'fold_{fold_idx}')
    os.makedirs(fold_log_dir, exist_ok=True)
    
    # Save fold split information
    split_info = {
        'fold': fold_idx,
        'train_size': len(fold_data['train_images']),
        'val_size': len(fold_data['val_images']),
        'train_indices': fold_data['train_indices'].tolist(),
        'val_indices': fold_data['val_indices'].tolist(),
    }
    with open(os.path.join(fold_log_dir, 'fold_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create datasets
    train_ds = ThioSDataset(
        fold_data['train_images'], 
        fold_data['train_labels'], 
        args, 
        training=True
    )
    val_ds = ThioSDataset(
        fold_data['val_images'], 
        fold_data['val_labels'], 
        args, 
        training=False
    )
    
    # Model hyperparameters
    model_kwargs = {
        'num_classes': args.get('num_classes', NUM_CLASSES),
        'input_channels': args.get('input_channels', 3),
        'pred_patch_size': args.get('pred_patch_size', (256, 256)),
        'batch_size': args.get('batch_size', 16),
        'lr': args.get('lr', 1e-4),
        'num_workers': args.get('num_workers', 4),
        'background_index': args.get('background_index', 1),
        'spatial_dims': args.get('spatial_dims', 2),
        'log_dir': fold_log_dir,
        'dropout': args.get('dropout', 0.3),
        'weight_decay': args.get('weight_decay', 0.01),
        'label_smoothing': args.get('label_smoothing', 0.0),
        'reduce_lr_patience': args.get('reduce_lr_patience', None),
        'reduce_lr_factor': args.get('reduce_lr_factor', 0.5),
        'encoder_name': args.get('encoder_name', 'efficientnet-b4'),
        'pretrained': args.get('pretrained', True),
        'use_attention': args.get('use_attention', False),
        # V12: Boundary-precision parameters
        'boundary_loss_weight': args.get('boundary_loss_weight', 0.0),
        'tversky_params': args.get('tversky_params', None),
        'focal_weight': args.get('focal_weight', 0.4),
        'tversky_weight': args.get('tversky_weight', 0.6),
    }
    
    # Create model
    model = ThioSUnet2D(train_ds, val_ds, **model_kwargs)
    
    # Callbacks
    early_stopping_patience = args.get('early_stopping_patience', 50)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(fold_log_dir, 'checkpoints'),
            filename=f'fold{fold_idx}-' + '{epoch:03d}-{val_iou:.4f}',
            monitor='val_iou',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val_iou',
            mode='max',
            patience=early_stopping_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Training strategy
    if cmd_args.gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = 'auto'
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=cmd_args.accelerator,
        devices=cmd_args.gpus,
        strategy=strategy,
        max_epochs=args.get('nb_epochs', 300),
        callbacks=callbacks,
        default_root_dir=fold_log_dir,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )
    
    # Train
    print(f"\nStarting training for Fold {fold_idx + 1}...")
    trainer.fit(model, ckpt_path=cmd_args.resume_from)
    
    # Get best validation metrics
    best_val_iou = callbacks[0].best_model_score.item() if callbacks[0].best_model_score else None
    
    print(f"\nFold {fold_idx + 1} completed!")
    print(f"  Best val IoU: {best_val_iou:.4f}" if best_val_iou else "  No best score recorded")
    print(f"  Best checkpoint: {callbacks[0].best_model_path}")
    
    return {
        'fold': fold_idx,
        'best_val_iou': best_val_iou,
        'best_checkpoint': callbacks[0].best_model_path,
    }


def train_kfold(cmd_args):
    """Main k-fold training function."""
    # Load configuration
    args = load_config(cmd_args.config_file)
    
    # Set up base logging directory
    base_log_dir = os.path.abspath(args.get('log_dir', './logs'))
    os.makedirs(base_log_dir, exist_ok=True)
    
    print("="*70)
    print("ThioS Multiclass Segmentation - K-Fold Cross-Validation Training")
    print("="*70)
    print(f"Config: {cmd_args.config_file}")
    print(f"Base log dir: {base_log_dir}")
    print(f"Number of folds: {cmd_args.n_folds}")
    print(f"GPUs: {cmd_args.gpus}")
    
    # Save config to log dir
    with open(os.path.join(base_log_dir, 'training_config.json'), 'w') as f:
        json.dump(args, f, indent=2)
    
    # Set determinism for reproducibility
    set_determinism(seed=42)
    
    # Prepare k-fold splits
    fold_splits = prepare_kfold_splits(args, cmd_args.n_folds)
    
    # Train each fold
    fold_results = []
    
    if cmd_args.specific_fold is not None:
        # Train only specific fold
        if 0 <= cmd_args.specific_fold < cmd_args.n_folds:
            fold_idx = cmd_args.specific_fold
            result = train_single_fold(
                fold_idx, 
                fold_splits[fold_idx], 
                args, 
                cmd_args, 
                base_log_dir
            )
            fold_results.append(result)
        else:
            print(f"ERROR: Invalid fold index {cmd_args.specific_fold}. Must be 0 to {cmd_args.n_folds-1}")
            sys.exit(1)
    else:
        # Train all folds
        for fold_idx, fold_data in enumerate(fold_splits):
            result = train_single_fold(
                fold_idx, 
                fold_data, 
                args, 
                cmd_args, 
                base_log_dir
            )
            fold_results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    valid_ious = [r['best_val_iou'] for r in fold_results if r['best_val_iou'] is not None]
    
    if valid_ious:
        mean_iou = np.mean(valid_ious)
        std_iou = np.std(valid_ious)
        
        print(f"\nValidation IoU across folds:")
        for result in fold_results:
            if result['best_val_iou'] is not None:
                print(f"  Fold {result['fold'] + 1}: {result['best_val_iou']:.4f}")
        
        print(f"\nMean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
        
        # Save summary
        summary = {
            'n_folds': cmd_args.n_folds,
            'mean_iou': float(mean_iou),
            'std_iou': float(std_iou),
            'fold_results': fold_results,
        }
        with open(os.path.join(base_log_dir, 'kfold_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {os.path.join(base_log_dir, 'kfold_summary.json')}")
    else:
        print("No valid IoU scores recorded.")
    
    print("="*70)
    
    return fold_results


def main():
    args = parse_args()
    train_kfold(args)


if __name__ == "__main__":
    main()
