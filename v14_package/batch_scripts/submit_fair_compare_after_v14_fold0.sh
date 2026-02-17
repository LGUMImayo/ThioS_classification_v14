#!/bin/bash
set -euo pipefail

BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/ThioS_classification"
LOG_DIR="${BASE_DIR}/logs/efficientnet_b4_v14_kfold"
TRIGGER_DIR="${LOG_DIR}/fold_1"
BATCH_SCRIPT="${BASE_DIR}/batch_scripts/run_fair_compare.sh"
STATE_FILE="${LOG_DIR}/.fair_compare_submitted_after_fold0"

if [[ -f "${STATE_FILE}" ]]; then
  echo "[$(date)] Fair comparison already submitted earlier: ${STATE_FILE}"
  exit 0
fi

echo "[$(date)] Waiting for V14 fold 0 completion trigger..."
echo "Trigger condition: directory exists -> ${TRIGGER_DIR}"

while true; do
  if [[ -d "${TRIGGER_DIR}" ]]; then
    echo "[$(date)] Detected fold_1 directory. Fold 0 is complete."
    submit_out=$(sbatch "${BATCH_SCRIPT}")
    echo "[$(date)] ${submit_out}"
    echo "${submit_out}" > "${STATE_FILE}"
    exit 0
  fi
  sleep 180
 done
