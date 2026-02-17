#!/bin/bash
#SBATCH --job-name=thios_v14
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/thios_v14_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/thios_v14_%j.err
#SBATCH --partition=gpu-n24-170g-4x-a100-40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=15-00:00:00

# ============================================================================
# ThioS V14 Training — Tight Boundaries with Corrected Labels
# ============================================================================
#
# GOAL: Achieve V5-like tight boundaries while using Erica's corrected (V11) labels.
#
# ROOT CAUSE: V13's expanded boundaries are NOT a model problem — the corrected
# labels have 1.9x more foreground area than the old labels. V13 is faithfully
# learning the corrected labels. V14 deliberately under-predicts foreground to
# produce conservative/tight boundaries.
#
# V13 → V14 changes:
#
# 1. HIGHER FP PENALTY: Tversky α=0.7/β=0.3 for ALL classes
#    Penalizes false positives 2.3x more than false negatives.
#    Model is incentivized to UNDER-predict rather than over-predict.
#    (V13 was α=0.5/β=0.5 balanced, V12 was α=0.6/β=0.4)
#
# 2. EQUAL CLASS WEIGHTS (all 1.0)
#    V13: bg=1.0, diffuse=1.5, plaque=1.5, tangle=2.0
#    V14: bg=1.0, diffuse=1.0, plaque=1.0, tangle=1.0
#    No extra incentive to predict rare foreground classes.
#
# 3. NO OVERSAMPLING (all factors = 1)
#    V13: tangle 2x
#    V14: none
#    Model sees natural class distribution → less foreground bias.
#
# 4. TVERSKY-ONLY LOSS (no focal)
#    V13: 0.5 focal + 0.5 tversky
#    V5 (tightest model) used Tversky-only. Focal loss focuses on hard
#    boundary pixels, pushing model to expand into uncertain edges.
#
# 5. CONFIDENCE THRESHOLDING (threshold=0.5)
#    During validation and inference, foreground class only assigned if
#    softmax probability > 0.5. Otherwise assigned to background.
#    Prevents low-confidence "bleed" at boundaries.
#
# Data: Same patches_v11 with Erica's corrected labels (27,997 patches)
# Architecture: Same EfficientNet-B4 + UNet decoder
# ============================================================================

set -e

echo "============================================================"
echo "ThioS V14 Training — Tight Boundaries — $(date)"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate ThioS_net

echo "Environment: ThioS_net"
echo "Python: $(which python)"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "MONAI: $(python -c 'import monai; print(monai.__version__)')"
echo "PL: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')"

nvidia-smi

BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/ThioS_classification"
SCRIPTS_DIR="${BASE_DIR}/scripts"
CONFIG_DIR="${BASE_DIR}/configs"

PATCHES_DIR="${BASE_DIR}/patches_v11"
LOGS_DIR="${BASE_DIR}/logs/efficientnet_b4_v14_kfold"

mkdir -p "${LOGS_DIR}"

echo ""
echo "============================================================"
echo "V14 Training (5-Fold Stratified Cross-Validation)"
echo "============================================================"
echo "Using patches from: ${PATCHES_DIR}"
echo "Log dir: ${LOGS_DIR}"
echo ""

TRAIN_CONFIG="${CONFIG_DIR}/setup-train_thios_efficientnet_v14.json"

echo "Training config: ${TRAIN_CONFIG}"
echo ""
echo "V14 Key changes from V13 (targeting tight boundaries):"
echo "  Tversky: α=0.7/β=0.3 HIGH FP PENALTY (was α=0.5/β=0.5)"
echo "  Class weights: ALL 1.0 — equal, no foreground boost"
echo "  Oversampling: NONE (was tangle 2x)"
echo "  Loss: Tversky-only (no focal) — matches V5's loss setup"
echo "  Confidence threshold: 0.5 (new — prevents low-conf boundary bleed)"
echo "  Boundary loss: NONE"
echo ""

# Run k-fold training
python "${SCRIPTS_DIR}/train_thios_kfold.py" \
    --config_file "${TRAIN_CONFIG}" \
    --n_folds 5 \
    --gpus 1 \
    --accelerator gpu

echo ""
echo "============================================================"
echo "Analyze K-Fold Results"
echo "============================================================"

python "${SCRIPTS_DIR}/analyze_kfold_results.py" \
    --log_dir "${LOGS_DIR}" \
    --n_folds 5

echo ""
echo "============================================================"
echo "ThioS V14 Training Complete — $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Checkpoints: ${LOGS_DIR}/fold_*/checkpoints/"
echo "  - Summary: ${LOGS_DIR}/kfold_summary.json"
echo "============================================================"
