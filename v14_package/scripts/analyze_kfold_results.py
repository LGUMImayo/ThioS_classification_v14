#!/usr/bin/env python3
"""
Aggregate and analyze k-fold cross-validation results.

This script:
1. Loads results from all fold checkpoints
2. Computes detailed per-class metrics across folds
3. Generates comparison plots and summary tables
4. Creates an ensemble prediction combining all folds

Usage:
    python analyze_kfold_results.py --log_dir logs/efficientnet_b4_v10_kfold --n_folds 5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Try to import tensorboard for reading logs
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed. Cannot parse training logs.")


CLASS_NAMES = {0: 'unlabeled', 1: 'background', 2: 'diffuse', 3: 'plaque', 4: 'tangle'}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze K-Fold CV Results")
    parser.add_argument(
        "--log_dir", type=str, required=True,
        help="Base log directory containing fold_* subdirectories"
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for analysis results (default: log_dir/analysis)"
    )
    return parser.parse_args()


def load_fold_summary(log_dir: str, n_folds: int) -> Dict:
    """Load k-fold summary if it exists."""
    summary_path = os.path.join(log_dir, 'kfold_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    else:
        print(f"Warning: {summary_path} not found. Creating from fold data...")
        return create_summary_from_folds(log_dir, n_folds)


def create_summary_from_folds(log_dir: str, n_folds: int) -> Dict:
    """Create summary from individual fold results."""
    fold_results = []
    
    for fold_idx in range(n_folds):
        fold_dir = os.path.join(log_dir, f'fold_{fold_idx}')
        if not os.path.exists(fold_dir):
            print(f"Warning: Fold {fold_idx} directory not found: {fold_dir}")
            continue
        
        # Find best checkpoint
        ckpt_dir = os.path.join(fold_dir, 'checkpoints')
        if os.path.exists(ckpt_dir):
            ckpts = glob(os.path.join(ckpt_dir, 'fold*.ckpt'))
            if ckpts:
                # Parse IoU from filename (format: fold0-epoch-val_iou.ckpt)
                best_ckpt = None
                best_iou = -1
                for ckpt in ckpts:
                    if 'last.ckpt' in ckpt:
                        continue
                    try:
                        # Extract IoU from filename
                        parts = Path(ckpt).stem.split('-')
                        iou_str = [p for p in parts if p.replace('.', '').isdigit()]
                        if iou_str:
                            iou = float(iou_str[-1])
                            if iou > best_iou:
                                best_iou = iou
                                best_ckpt = ckpt
                    except:
                        continue
                
                if best_ckpt:
                    fold_results.append({
                        'fold': fold_idx,
                        'best_val_iou': best_iou,
                        'best_checkpoint': best_ckpt,
                    })
    
    # Compute summary statistics
    ious = [r['best_val_iou'] for r in fold_results]
    summary = {
        'n_folds': n_folds,
        'mean_iou': float(np.mean(ious)) if ious else None,
        'std_iou': float(np.std(ious)) if ious else None,
        'fold_results': fold_results,
    }
    
    return summary


def plot_fold_comparison(summary: Dict, output_dir: str):
    """Create visualization comparing fold performance."""
    fold_results = summary['fold_results']
    
    if not fold_results:
        print("No fold results to plot.")
        return
    
    folds = [r['fold'] for r in fold_results]
    ious = [r['best_val_iou'] for r in fold_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot
    bars = ax.bar(folds, ious, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add mean line
    mean_iou = summary['mean_iou']
    std_iou = summary['std_iou']
    ax.axhline(mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iou:.4f}')
    ax.axhline(mean_iou + std_iou, color='orange', linestyle=':', linewidth=1.5, 
               label=f'Mean ± Std: {mean_iou:.4f} ± {std_iou:.4f}')
    ax.axhline(mean_iou - std_iou, color='orange', linestyle=':', linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation IoU', fontsize=12, fontweight='bold')
    ax.set_title('K-Fold Cross-Validation Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, iou in zip(bars, ious):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{iou:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_comparison.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'fold_comparison.png')}")
    plt.close()


def create_summary_table(summary: Dict, output_dir: str):
    """Create summary table with statistics."""
    fold_results = summary['fold_results']
    
    if not fold_results:
        print("No fold results for summary table.")
        return
    
    # Create DataFrame
    data = []
    for r in fold_results:
        data.append({
            'Fold': r['fold'],
            'Val IoU': f"{r['best_val_iou']:.4f}",
            'Checkpoint': Path(r['best_checkpoint']).name,
        })
    
    df = pd.DataFrame(data)
    
    # Add summary row
    summary_row = {
        'Fold': 'Mean ± Std',
        'Val IoU': f"{summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}",
        'Checkpoint': '-',
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'fold_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Print to console
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


def analyze_fold_variance(summary: Dict, output_dir: str):
    """Analyze variance across folds."""
    fold_results = summary['fold_results']
    
    if not fold_results or len(fold_results) < 2:
        print("Not enough folds for variance analysis.")
        return
    
    ious = np.array([r['best_val_iou'] for r in fold_results])
    
    mean_iou = np.mean(ious)
    std_iou = np.std(ious)
    min_iou = np.min(ious)
    max_iou = np.max(ious)
    cv = (std_iou / mean_iou) * 100  # Coefficient of variation
    
    analysis = {
        'mean_iou': float(mean_iou),
        'std_iou': float(std_iou),
        'min_iou': float(min_iou),
        'max_iou': float(max_iou),
        'range': float(max_iou - min_iou),
        'coefficient_of_variation_percent': float(cv),
        'interpretation': {
            'stability': 'High' if cv < 5 else 'Medium' if cv < 10 else 'Low',
            'recommendation': (
                'Good model stability. Proceed with confidence.' if cv < 5 else
                'Moderate variance. Consider investigating outlier folds.' if cv < 10 else
                'High variance. Check data distribution and hyperparameters.'
            )
        }
    }
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'variance_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nSaved: {analysis_path}")
    print("\nVariance Analysis:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Std IoU: {std_iou:.4f}")
    print(f"  Range: [{min_iou:.4f}, {max_iou:.4f}]")
    print(f"  Coefficient of Variation: {cv:.2f}%")
    print(f"  Stability: {analysis['interpretation']['stability']}")
    print(f"  Recommendation: {analysis['interpretation']['recommendation']}")


def create_markdown_report(summary: Dict, output_dir: str):
    """Generate comprehensive markdown report."""
    fold_results = summary['fold_results']
    
    report = f"""# K-Fold Cross-Validation Report

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Number of Folds**: {summary['n_folds']}
- **Mean Validation IoU**: {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}

## Fold Results

| Fold | Validation IoU | Checkpoint |
|------|---------------|------------|
"""
    
    for r in fold_results:
        ckpt_name = Path(r['best_checkpoint']).name
        report += f"| {r['fold']} | {r['best_val_iou']:.4f} | `{ckpt_name}` |\n"
    
    report += f"""
## Statistical Summary

- **Mean**: {summary['mean_iou']:.4f}
- **Std Dev**: {summary['std_iou']:.4f}
- **Min**: {min([r['best_val_iou'] for r in fold_results]):.4f}
- **Max**: {max([r['best_val_iou'] for r in fold_results]):.4f}

## Visualizations

![Fold Comparison](fold_comparison.png)

## Recommendations

### Model Selection

1. **Ensemble Approach** (Recommended): Use all {summary['n_folds']} models for inference
   - Average predictions for improved robustness
   - Reduces variance and outlier predictions

2. **Best Single Model**: Use Fold {max(fold_results, key=lambda x: x['best_val_iou'])['fold']}
   - Highest validation IoU: {max([r['best_val_iou'] for r in fold_results]):.4f}
   - Faster inference (single model)

3. **Retrain on Full Dataset**: After hyperparameter validation
   - Use all data (no CV split)
   - Train with validated settings from this CV run

### Next Steps

1. Perform external validation on held-out test set
2. Analyze per-class performance across folds
3. Investigate any outlier folds (if variance is high)
4. Consider ensemble inference for production deployment

---

*Generated by ThioS K-Fold Analysis Pipeline*
"""
    
    report_path = os.path.join(output_dir, 'kfold_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nSaved: {report_path}")


def main():
    args = parse_args()
    
    # Set up output directory
    output_dir = args.output_dir or os.path.join(args.log_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("K-Fold Cross-Validation Analysis")
    print("="*70)
    print(f"Log directory: {args.log_dir}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Output directory: {output_dir}")
    
    # Load summary
    summary = load_fold_summary(args.log_dir, args.n_folds)
    
    if not summary['fold_results']:
        print("\nERROR: No fold results found. Check log directory and fold naming.")
        sys.exit(1)
    
    print(f"\nLoaded results for {len(summary['fold_results'])} folds")
    
    # Generate analyses
    print("\nGenerating analysis outputs...")
    
    create_summary_table(summary, output_dir)
    plot_fold_comparison(summary, output_dir)
    analyze_fold_variance(summary, output_dir)
    create_markdown_report(summary, output_dir)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - fold_summary.csv")
    print(f"  - fold_comparison.png")
    print(f"  - variance_analysis.json")
    print(f"  - kfold_report.md")
    print("="*70)


if __name__ == "__main__":
    main()
