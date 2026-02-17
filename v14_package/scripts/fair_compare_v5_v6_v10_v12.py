#!/usr/bin/env python3
"""
Fair Comparison: V5, V6, V10, V11, V12, V13, V14

Extracts regions from full-resolution source images and runs all models
on the SAME spatial regions at their native 128x128 resolution.

Ground truth labels are the fixed/corrected labels (processed_data_v11).
Output: side-by-side visualization similar to the old V3/V5/V6/V7/V8 comparison.
"""

import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
import monai.utils.enums
torch.serialization.add_safe_globals([monai.utils.enums.TraceKeys])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from unet2D_thios import ThioSUnet2D

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/ThioS_classification"
FULL_IMAGES_DIR = os.path.join(BASE_DIR, "processed_data_v11", "images")
FULL_LABELS_DIR = os.path.join(BASE_DIR, "processed_data_v11", "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "fair_comparison_v5_v6_v10_v11_v12_v13_v14_fixedlabels")

# Checkpoints per version (all use 128x128)
VERSIONS = {
    "V5": {
        "ckpt": os.path.join(BASE_DIR, "logs/efficientnet_b4_v5/checkpoints/thios-epoch=093-val_iou=0.3635.ckpt"),
        "val_iou": 0.3635,
        "notes": "Balanced Tversky α=0.5/β=0.5, class weights",
    },
    "V6": {
        "ckpt": os.path.join(BASE_DIR, "logs/efficientnet_b4_v6/checkpoints/thios-epoch=124-val_iou=0.3628.ckpt"),
        "val_iou": 0.3628,
        "notes": "Class-specific Tversky, oversampling",
    },
    "V10": {
        "ckpt": None,
        "val_iou": None,
        "notes": "5-fold, balanced Tversky, corrected labels",
    },
    "V11": {
        "ckpt": None,
        "val_iou": None,
        "notes": "5-fold, corrected labels (Erica fixes)",
    },
    "V12": {
        "ckpt": None,
        "val_iou": None,
        "notes": "Boundary-precise: Tversky α=0.6/β=0.4, boundary loss",
    },
    "V13": {
        "ckpt": None,
        "val_iou": None,
        "notes": "Conservative: balanced Tversky α=0.5/β=0.5, no boundary loss, corrected labels",
    },
    "V14": {
        "ckpt": None,
        "val_iou": None,
        "notes": "Tight-boundary variant: FP-penalized Tversky + confidence thresholding",
    },
}

def auto_discover_best_fold0_ckpt(log_subdir):
    """Find best fold_0 checkpoint by val_iou from a k-fold log directory."""
    ckpt_dir = os.path.join(BASE_DIR, "logs", log_subdir, "fold_0", "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "fold0-epoch=*-val_iou=*.ckpt"))
    if not ckpts:
        return None, None
    best = max(ckpts, key=lambda p: float(p.split("val_iou=")[1].replace(".ckpt", "")))
    best_iou = float(best.split("val_iou=")[1].replace(".ckpt", ""))
    return best, best_iou


# Auto-discover best fold 0 checkpoints for all multi-fold versions
KFOLD_LOG_SUBDIR = {
    "V10": "efficientnet_b4_v10_kfold",
    "V11": "efficientnet_b4_v11_kfold",
    "V12": "efficientnet_b4_v12_kfold",
    "V13": "efficientnet_b4_v13_kfold",
    "V14": "efficientnet_b4_v14_kfold",
}

for version_name, log_subdir in KFOLD_LOG_SUBDIR.items():
    ckpt, best_iou = auto_discover_best_fold0_ckpt(log_subdir)
    VERSIONS[version_name]["ckpt"] = ckpt
    VERSIONS[version_name]["val_iou"] = best_iou

NUM_CLASSES = 5
CLASS_NAMES = {0: 'unlabeled', 1: 'background', 2: 'diffuse', 3: 'plaque', 4: 'tangle'}

# Colors matching the old visualization (tangle = cyan)
CLASS_COLORS = {
    0: [0, 0, 0],           # unlabeled - black
    1: [128, 128, 128],     # background - gray
    2: [255, 255, 0],       # diffuse - yellow
    3: [255, 0, 0],         # plaque - red
    4: [0, 255, 255],       # tangle - cyan
}

# Specific regions from the old V3/V5/V6/V7/V8 fair comparison
# These are the EXACT coordinates used in the original visualization
PREDEFINED_REGIONS = [
    ('2025_12_05_NA24_-019_A_MCF_Neuropath', 1291, 9044),
    ('2025_12_05_NA24_-019_B_MCF_Neuropath', 4386, 1084),
    ('2025_12_05_NA24_-112_A_MCF_Neuropath', 6304, 6252),
    ('2025_12_05_NA24_-112_B_MCF_Neuropath', 8837, 8689),
]

MODEL_PARAMS = {
    'num_classes': NUM_CLASSES,
    'input_channels': 3,
    'pred_patch_size': [128, 128],
    'batch_size': 64,
    'lr': 0.00005,
    'num_workers': 0,
    'background_index': 1,
    'spatial_dims': 2,
    'log_dir': '/tmp',
    'encoder_name': 'efficientnet-b4',
    'pretrained': False,
    'contrast_enhance': False,
    'early_stopping_patience': 50,
    'image_dir': '/tmp',
    'label_dir': '/tmp',
    'weight_dir': '/tmp',
    'patch_size': [128, 128],
}


def colorize_mask(mask):
    """Convert label mask to RGB using CLASS_COLORS."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        rgb[mask == c] = color
    return rgb


def load_model(ckpt_path, device):
    """Load model from checkpoint."""
    model = ThioSUnet2D(train_ds=None, val_ds=None, **MODEL_PARAMS)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_patch(patch):
    """Preprocess image patch for inference (percentile normalization)."""
    lower = np.percentile(patch, 1.0)
    upper = np.percentile(patch, 99.5)
    patch = np.clip(patch, lower, upper)
    patch = (patch - lower) / (upper - lower + 1e-8)
    patch = patch.astype(np.float32)
    
    if patch.ndim == 3:
        patch = np.transpose(patch, (2, 0, 1))  # HWC -> CHW
    else:
        patch = patch[None, :, :]
    
    return torch.from_numpy(patch).unsqueeze(0)


def predict(model, image_np, device):
    """Run inference on a single image."""
    input_tensor = preprocess_patch(image_np).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        logits = model.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return pred


def compute_iou_labeled(label, pred):
    """Compute per-class IoU on labeled pixels only."""
    labeled_mask = (label > 0)
    ious = {}
    for c in range(1, NUM_CLASSES):
        gt_c = np.logical_and(label == c, labeled_mask)
        pred_c = np.logical_and(pred == c, labeled_mask)
        intersection = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        if union > 0:
            ious[CLASS_NAMES[c]] = intersection / union
    return ious


def compute_dice_labeled(label, pred):
    """Compute per-class Dice coefficient on labeled pixels only."""
    labeled_mask = (label > 0)
    dices = {}
    for c in range(1, NUM_CLASSES):
        gt_c = np.logical_and(label == c, labeled_mask)
        pred_c = np.logical_and(pred == c, labeled_mask)
        intersection = np.logical_and(gt_c, pred_c).sum()
        denom = gt_c.sum() + pred_c.sum()
        if denom > 0:
            dices[CLASS_NAMES[c]] = 2 * intersection / denom
    return dices


def extract_boundary(mask, width=2):
    """Extract boundary pixels from a binary mask."""
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask, iterations=width)
    boundary = mask & ~eroded
    return boundary


def compute_boundary_f1(label, pred, width=2):
    """
    Compute boundary F1 score - measures how well boundaries align.
    Higher = better boundary precision.
    """
    labeled_mask = (label > 0)
    boundary_f1s = {}
    
    for c in range(1, NUM_CLASSES):
        gt_c = np.logical_and(label == c, labeled_mask)
        pred_c = np.logical_and(pred == c, labeled_mask)
        
        if gt_c.sum() == 0:
            continue
        
        # Extract boundaries
        gt_boundary = extract_boundary(gt_c, width)
        pred_boundary = extract_boundary(pred_c, width)
        
        if gt_boundary.sum() == 0 and pred_boundary.sum() == 0:
            boundary_f1s[CLASS_NAMES[c]] = 1.0  # Perfect if no boundaries
            continue
        
        # Boundary precision: pred boundary pixels that match GT boundary
        # Boundary recall: GT boundary pixels that match pred boundary
        # Use distance tolerance
        from scipy.ndimage import distance_transform_edt
        
        if gt_boundary.sum() > 0:
            gt_dist = distance_transform_edt(~gt_boundary)
            pred_on_gt = pred_boundary & (gt_dist <= width)
            precision = pred_on_gt.sum() / max(pred_boundary.sum(), 1)
        else:
            precision = 0
        
        if pred_boundary.sum() > 0:
            pred_dist = distance_transform_edt(~pred_boundary)
            gt_on_pred = gt_boundary & (pred_dist <= width)
            recall = gt_on_pred.sum() / max(gt_boundary.sum(), 1)
        else:
            recall = 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        boundary_f1s[CLASS_NAMES[c]] = f1
    
    return boundary_f1s


def compute_all_metrics(label, pred):
    """Compute IoU, Dice, and Boundary F1 for all classes."""
    return {
        'iou': compute_iou_labeled(label, pred),
        'dice': compute_dice_labeled(label, pred),
        'boundary_f1': compute_boundary_f1(label, pred),
    }


def extract_test_regions(num_samples=10, seed=42, use_predefined=True):
    """
    Extract test regions from full-resolution images.
    If use_predefined=True, uses the exact same coordinates as the old V3/V5/V6/V7/V8 comparison.
    Returns list of dicts with 128x128 image/label pairs and metadata.
    """
    random.seed(seed)
    test_regions = []
    
    if not os.path.exists(FULL_IMAGES_DIR):
        print(f"Error: {FULL_IMAGES_DIR} not found")
        return test_regions
    
    # Use predefined regions from the old comparison
    if use_predefined:
        for img_base, y, x in PREDEFINED_REGIONS:
            # Find the actual image file
            img_file = None
            for ext in ['.tiff', '.tif']:
                candidate = img_base + ext
                if os.path.exists(os.path.join(FULL_IMAGES_DIR, candidate)):
                    img_file = candidate
                    break
            
            if not img_file:
                print(f"  Warning: Image not found for {img_base}")
                continue
            
            img_path = os.path.join(FULL_IMAGES_DIR, img_file)
            label_file = img_base + '.png'
            label_path = os.path.join(FULL_LABELS_DIR, label_file)
            
            if not os.path.exists(label_path):
                print(f"  Warning: Label not found: {label_file}")
                continue
            
            try:
                full_img = np.array(Image.open(img_path))
                full_label = np.array(Image.open(label_path))
                
                region_size = 128
                img_region = full_img[y:y+region_size, x:x+region_size]
                label_region = full_label[y:y+region_size, x:x+region_size]
                
                name = f"{img_base}_y{y}_x{x}"
                test_regions.append({
                    'image': img_region,
                    'label': label_region,
                    'name': name,
                    'source': img_file,
                    'coords': (y, x),
                })
                print(f"  Loaded predefined region from {img_file} at ({y}, {x})")
                
            except Exception as e:
                print(f"  Warning: Could not load {img_file}: {e}")
                continue
        
        return test_regions
    
    # Fall back to random sampling
    image_files = sorted([f for f in os.listdir(FULL_IMAGES_DIR) if f.endswith(('.tiff', '.tif'))])
    
    for img_file in image_files:
        if len(test_regions) >= num_samples:
            break
        
        img_path = os.path.join(FULL_IMAGES_DIR, img_file)
        label_file = img_file.replace('.tiff', '.png').replace('.tif', '.png')
        label_path = os.path.join(FULL_LABELS_DIR, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        try:
            full_img = np.array(Image.open(img_path))
            full_label = np.array(Image.open(label_path))
            h, w = full_img.shape[:2]
            
            # Sample regions with biomarkers
            region_size = 128
            attempts = 0
            max_attempts = 100
            
            while attempts < max_attempts and len(test_regions) < num_samples:
                y = random.randint(0, max(0, h - region_size))
                x = random.randint(0, max(0, w - region_size))
                
                label_region = full_label[y:y+region_size, x:x+region_size]
                
                # Check for biomarkers (classes 2, 3, 4)
                biomarker_mask = (label_region > 1)
                biomarker_ratio = np.sum(biomarker_mask) / label_region.size
                
                if biomarker_ratio >= 0.01:  # At least 1% biomarkers
                    img_region = full_img[y:y+region_size, x:x+region_size]
                    
                    name = f"{os.path.splitext(img_file)[0]}_y{y}_x{x}"
                    test_regions.append({
                        'image': img_region,
                        'label': label_region,
                        'name': name,
                        'source': img_file,
                        'coords': (y, x),
                    })
                    print(f"  Extracted region from {img_file} at ({y}, {x})")
                    break
                
                attempts += 1
                
        except Exception as e:
            print(f"  Warning: Could not process {img_file}: {e}")
            continue
    
    return test_regions


def visualize_fair_comparison(models, test_regions, output_dir, device):
    """
    Generate fair side-by-side predictions from all models.
    Layout: Original | GT | V5 | V6 | V10 | V12
    Shows IoU, Dice, and Boundary F1 metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    version_names = list(models.keys())
    n_cols = 2 + len(version_names)  # Original + GT + versions
    
    all_results = []
    
    for idx, region in enumerate(tqdm(test_regions, desc="Generating comparisons")):
        image = region['image']
        label = region['label']
        name = region['name']
        
        # Run all models and compute all metrics
        predictions = {}
        all_metrics = {}
        for vname, model in models.items():
            pred = predict(model, image, device)
            predictions[vname] = pred
            all_metrics[vname] = compute_all_metrics(label, pred)
        
        # Create visualization
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
        
        # Original image
        if image.ndim == 3:
            display_img = image / max(image.max(), 1)
        else:
            display_img = np.stack([image] * 3, axis=-1)
            display_img = display_img / max(display_img.max(), 1)
        
        axes[0].imshow(display_img)
        axes[0].set_title('Original (128×128)', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth
        gt_rgb = colorize_mask(label)
        axes[1].imshow(gt_rgb)
        labeled_pct = 100 * np.sum(label > 0) / label.size
        axes[1].set_title(f'Ground Truth\n({labeled_pct:.0f}% labeled)', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        # Model predictions with multiple metrics
        for i, vname in enumerate(version_names):
            pred_rgb = colorize_mask(predictions[vname])
            axes[2 + i].imshow(pred_rgb)
            
            metrics = all_metrics[vname]
            miou = np.mean(list(metrics['iou'].values())) if metrics['iou'] else 0
            mdice = np.mean(list(metrics['dice'].values())) if metrics['dice'] else 0
            mbf1 = np.mean(list(metrics['boundary_f1'].values())) if metrics['boundary_f1'] else 0
            
            # Show all 3 metrics in title
            axes[2 + i].set_title(f'{vname}\nIoU={miou:.3f} Dice={mdice:.3f}\nBndF1={mbf1:.3f}', 
                                  fontsize=10, fontweight='bold')
            axes[2 + i].axis('off')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=np.array(CLASS_COLORS[1])/255, label='Background'),
            mpatches.Patch(color=np.array(CLASS_COLORS[2])/255, label='Diffuse'),
            mpatches.Patch(color=np.array(CLASS_COLORS[3])/255, label='Plaque'),
            mpatches.Patch(color=np.array(CLASS_COLORS[4])/255, label='Tangle'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, frameon=True)
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        out_path = os.path.join(output_dir, f'fair_comparison_{idx:03d}_{name}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Store results with all metrics
        result = {'idx': idx, 'name': name, 'source': region['source']}
        for vname in version_names:
            metrics = all_metrics[vname]
            result[f'{vname}_iou'] = np.mean(list(metrics['iou'].values())) if metrics['iou'] else 0
            result[f'{vname}_dice'] = np.mean(list(metrics['dice'].values())) if metrics['dice'] else 0
            result[f'{vname}_bf1'] = np.mean(list(metrics['boundary_f1'].values())) if metrics['boundary_f1'] else 0
        all_results.append(result)
        
        # Print per-region metrics
        print(f"  Region {idx}:")
        for vname in version_names:
            iou = result[f'{vname}_iou']
            dice = result[f'{vname}_dice']
            bf1 = result[f'{vname}_bf1']
            print(f"    {vname}: IoU={iou:.3f}  Dice={dice:.3f}  BndF1={bf1:.3f}")
    
    return all_results


def print_summary(all_results, version_names):
    """Print summary tables for IoU, Dice, and Boundary F1."""
    
    # Print separate tables for each metric
    for metric_name, metric_key in [('IoU', 'iou'), ('DICE', 'dice'), ('BOUNDARY F1', 'bf1')]:
        print("\n" + "=" * 100)
        print(f"FAIR COMPARISON — {metric_name} — {', '.join(version_names)} (same spatial regions, 128×128)")
        print("=" * 100)
        
        # Header
        header = f"{'Region':40s}"
        for vname in version_names:
            header += f" | {vname:>8s}"
        print(header)
        print("-" * len(header))
        
        for r in all_results:
            row = f"{r['name'][:40]:40s}"
            vals = [r.get(f'{v}_{metric_key}', 0) for v in version_names]
            best_val = max(vals) if vals else 0
            for vname in version_names:
                val = r.get(f'{vname}_{metric_key}', 0)
                marker = " *" if val == best_val and val > 0 else "  "
                row += f" | {val:>6.3f}{marker}"
            print(row)
        
        # Averages
        print("-" * len(header))
        avg_row = f"{'AVERAGE':40s}"
        for vname in version_names:
            vals = [r.get(f'{vname}_{metric_key}', 0) for r in all_results]
            avg_row += f" | {np.mean(vals):>6.3f}  "
        print(avg_row)
    
    # Summary comparison
    print("\n" + "=" * 100)
    print("SUMMARY — Average across all regions")
    print("=" * 100)
    print(f"{'Metric':15s}", end="")
    for vname in version_names:
        print(f" | {vname:>8s}", end="")
    print()
    print("-" * 60)
    
    for metric_name, metric_key in [('IoU', 'iou'), ('Dice', 'dice'), ('Boundary F1', 'bf1')]:
        row = f"{metric_name:15s}"
        avgs = []
        for vname in version_names:
            vals = [r.get(f'{vname}_{metric_key}', 0) for r in all_results]
            avg = np.mean(vals)
            avgs.append(avg)
        best_avg = max(avgs)
        for i, vname in enumerate(version_names):
            marker = " *" if avgs[i] == best_avg else "  "
            row += f" | {avgs[i]:>6.3f}{marker}"
        print(row)


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_predefined', action='store_true', default=True,
                        help='Use predefined regions from old V3/V5/V6/V7/V8 comparison')
    parser.add_argument('--random', action='store_true',
                        help='Use random sampling instead of predefined regions')
    args = parser.parse_args()
    
    # If --random is specified, disable predefined
    use_predefined = args.use_predefined and not args.random
    
    if args.output_dir:
        OUTPUT_DIR = os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load models
    print("Loading models...")
    models = {}
    for vname, vinfo in VERSIONS.items():
        ckpt = vinfo["ckpt"]
        if ckpt and os.path.exists(ckpt):
            print(f"  {vname}: {os.path.basename(ckpt)} (val_iou={vinfo['val_iou']})")
            models[vname] = load_model(ckpt, device)
        else:
            print(f"  SKIP {vname}: checkpoint not found")
    
    print(f"\nLoaded {len(models)} models: {list(models.keys())}\n")
    
    # Extract test regions
    if use_predefined:
        print("Using PREDEFINED regions (same as old V3/V5/V6/V7/V8 comparison)...")
    else:
        print(f"Extracting {args.num_samples} random test regions...")
    test_regions = extract_test_regions(num_samples=args.num_samples, seed=args.seed, use_predefined=use_predefined)
    print(f"Found {len(test_regions)} test regions\n")
    
    if not test_regions:
        print("Error: No test regions found. Exiting.")
        return
    
    # Generate visualizations
    print("Generating fair comparison visualizations...")
    version_names = list(models.keys())
    all_results = visualize_fair_comparison(models, test_regions, OUTPUT_DIR, device)
    
    # Print summary
    print_summary(all_results, version_names)
    
    # Version info
    print(f"\n  Versions loaded:")
    for vname, vinfo in VERSIONS.items():
        if vname in models:
            print(f"    {vname}: val_iou={vinfo['val_iou']:.4f} — {vinfo['notes']}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  All models run on same 128×128 regions from processed_data_v11 (fixed labels)")


if __name__ == "__main__":
    main()
