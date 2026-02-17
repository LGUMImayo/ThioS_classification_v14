# ThioS V14 Segmentation Package

## Overview

This package contains the V14 code snapshot for multiclass ThioS semantic
segmentation (pixel-wise classification):

| Class ID | Name       | Description                     |
|----------|------------|---------------------------------|
| 0        | Unlabeled  | No annotation (ignored in loss) |
| 1        | Background | Labeled background tissue       |
| 2        | Diffuse    | Diffuse amyloid plaques         |
| 3        | Plaque     | Neuritic/cored plaques          |
| 4        | Tangle     | Neurofibrillary tangles         |

V14 goal: produce tighter, more conservative boundaries while using corrected
labels.

---

## Workflow:

```
Input (128×128×3)
    ↓
EfficientNet-B4 Encoder (pretrained, extracts features at multiple scales)
    ↓
UNet Decoder (upsamples back to full resolution)
    ↓
Segmentation Head (1×1 conv → 128×128×5)
    ↓
Softmax → per-pixel probabilities for 5 classes
    ↓
Argmax → per-pixel predicted class (bg/unlabeled/diffuse/plaque/tangle)
```
---

## What's in this package

```
v14_package/
├── README.md
├── scripts/
│   ├── unet2D_thios.py
│   ├── train_thios_kfold.py
│   ├── analyze_kfold_results.py
│   ├── prepare_thios_data_v2.py
│   ├── prepare_thios_patches.py
│   ├── generate_thios_weight_maps.py
│   └── fair_compare_v5_v6_v10_v12.py
├── configs/
│   ├── setup-train_thios_efficientnet_v14.json
│   └── patch_extraction_v11.json
├── batch_scripts/
│   ├── run_full_pipeline_v14.sh
│   ├── run_fair_compare.sh
│   └── submit_fair_compare_after_v14_fold0.sh
└── environment/
    └── environment_thios_net.yml
```

This is code-only (no large datasets/checkpoints bundled).

---

## V14 training settings (key differences)

From `configs/setup-train_thios_efficientnet_v14.json`:

- Loss composition: **Tversky-only**
  - `focal_weight: 0.0`
  - `tversky_weight: 1.0`
- FP-penalized Tversky:
  - `alpha=0.7, beta=0.3` for classes 1-4
- Class weights:
  - background=1.0, diffuse=1.2, plaque=1.0, tangle=1.5
- Boundary loss: disabled (`boundary_loss_weight: 0.0`)
- Oversampling factors: none (`{"4": 1, "2": 1, "3": 1}`)
- Confidence thresholding: enabled (`confidence_threshold: 0.5`)
  - applied in validation and prediction paths

---

## Architecture

- Encoder: EfficientNet-B4
- Decoder: MONAI FlexibleUNet
- Input: 128×128 pseudo-RGB patches
- Output: 5-channel per-pixel logits (semantic segmentation)

---

## Environment setup

```bash
conda env create -f environment/environment_thios_net.yml
conda activate ThioS_net
```

---

## Training (5-fold CV)

Use the V14 batch script:

```bash
sbatch batch_scripts/run_full_pipeline_v14.sh
```

Or run manually:

```bash
python scripts/train_thios_kfold.py \
  --config_file configs/setup-train_thios_efficientnet_v14.json \
  --n_folds 5 \
  --gpus 1 \
  --accelerator gpu
```

Results are written to:

- `.../logs/efficientnet_b4_v14_kfold/fold_*/checkpoints/`
- `.../logs/efficientnet_b4_v14_kfold/kfold_summary.json`

Early stopping is enabled in training code (monitor: `val_iou`, patience from
config).

---

## Recommended monitoring metrics for V14

Because V14 is intentionally conservative, do not rely on global accuracy alone.
Track:

- `val_iou_diffuse`, `val_iou_plaque`, `val_iou_tangle`
- `val_recall`
- `val_rare_iou_mean` (added)
- `val_rare_recall_mean` (added)
- `val_pred_frac_diffuse/plaque/tangle` (added)

A bad pattern is high `val_acc` but falling rare-class IoU/recall.

---

## Fair comparison workflow (fixed labels, same coordinates)

The comparison script is configured to:

- Evaluate on fixed labels (`processed_data_v11`)
- Use same predefined coordinates as prior tests
- Compare V5, V6, V10, V11, V12, V13, V14
- Use fold-0 best checkpoint for k-fold models

Run:

```bash
sbatch batch_scripts/run_fair_compare.sh
```

Auto-submit after V14 fold 0 starts fold 1:

```bash
nohup batch_scripts/submit_fair_compare_after_v14_fold0.sh > /path/to/log 2>&1 &
```

---

## Data path assumptions

Scripts/batch files assume the original base path:

`/fslustre/qhs/ext_chen_yuheng_mayo_edu/ThioS_classification`

If moving to another machine/project root, update absolute paths in:

- `configs/*.json`
- `batch_scripts/*.sh`
- `scripts/fair_compare_v5_v6_v10_v12.py`

---

## Hand-off checklist 

1. Create/activate `ThioS_net` env
2. Verify base paths in configs and batch scripts
3. Confirm data exists (new/fixed labels):
   - `patches_v11/images`, `patches_v11/labels`, `patches_v11/weights`
4. Launch V14 training (`run_full_pipeline_v14.sh`)
5. Monitor with TensorBoard
6. Run fair comparison after fold 0 with fixed labels

---

## Notes

- This package is intended as a reproducible V14 code snapshot for continue work

