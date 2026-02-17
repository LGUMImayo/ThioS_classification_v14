#!/bin/bash
#SBATCH --job-name=thios_fair_cmp
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/thios_fair_compare_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/thios_fair_compare_%j.err
#SBATCH --partition=gpu-n12-85g-1x-a100-40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00

# ============================================================================
# Fair Comparison: V5, V6, V10, V11, V12, V13, V14
# Runs all models on SAME predefined regions (same images + coordinates as previous tests)
# Evaluates against fixed/corrected labels (processed_data_v11)
# Output: fair_comparison_v5_v6_v10_v11_v12_v13_v14_fixedlabels/
# ============================================================================

echo "=== Fair Comparison: V5, V6, V10, V11, V12, V13, V14 (fixed labels) ==="
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate ThioS_net

echo "Python: $(which python)"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

cd /fslustre/qhs/ext_chen_yuheng_mayo_edu/ThioS_classification

python scripts/fair_compare_v5_v6_v10_v12.py \
    --use_predefined \
    --seed 42 \
    --output_dir fair_comparison_v5_v6_v10_v11_v12_v13_v14_fixedlabels

echo ""
echo "=== Completed: $(date) ==="
