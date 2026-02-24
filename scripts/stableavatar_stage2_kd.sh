#!/bin/bash
# Stage 2: CausalKD training for StableAvatar
#
# Trains the causal student on pre-computed ODE pairs (from Stage 1) via MSE.
# After training completes, extracts student net weights for Stage 3.
#
# Usage:
#   bash scripts/stableavatar_stage2_kd.sh [EXTRA_OPTS...]
#
# Examples:
#   bash scripts/stableavatar_stage2_kd.sh
#   bash scripts/stableavatar_stage2_kd.sh trainer.max_iter=20000
#
# Environment variables:
#   NGPUS                   Number of GPUs (auto-detected)
#   STABLEAVATAR_DATA_DIR   Training data with ODE pairs (default in config)
#   WANDB_PROJECT           WandB project name (default: stableavatar)
#   WANDB_API_KEY           WandB API key
#   RUN_NAME                Run name (default: kd_causal_<timestamp>)
#   SKIP_EXTRACT            Set to 1 to skip weight extraction after training

set -euo pipefail

export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-"FASTGEN_OUTPUT"}
export CKPT_ROOT_DIR=${CKPT_ROOT_DIR:-"${FASTGEN_OUTPUT_ROOT}/MODEL"}

NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}
RUN_NAME=${RUN_NAME:-"kd_causal_$(date +%y%m%d_%H%M)"}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}

echo "=== Stage 2: CausalKD Training ==="
echo "  GPUs:     ${NGPUS}"
echo "  Run name: ${RUN_NAME}"
echo ""

PYTHONPATH=$(pwd) torchrun \
    --nproc_per_node="${NGPUS}" \
    --standalone \
    train.py \
    --config=fastgen/configs/experiments/StableAvatar/config_kd_causal.py \
    - trainer.fsdp=True \
    log_config.name="${RUN_NAME}" \
    "$@"

echo "=== Stage 2 Training Complete ==="

# --- Bridge: extract student net weights ---
if [ "${SKIP_EXTRACT}" = "1" ]; then
    echo "Skipping weight extraction (SKIP_EXTRACT=1)."
    exit 0
fi

CKPT_DIR="${FASTGEN_OUTPUT_ROOT}/stableavatar/stableavatar_kd_causal/${RUN_NAME}/checkpoints"
LATEST_CKPT=$(ls -t "${CKPT_DIR}"/*.pth 2>/dev/null | head -1 || true)

if [ -z "${LATEST_CKPT}" ]; then
    echo "WARNING: No checkpoint found in ${CKPT_DIR}. Skipping weight extraction."
    exit 0
fi

OUTPUT_PT="${CKPT_ROOT_DIR}/stableavatar/ode_init.pt"
mkdir -p "$(dirname "${OUTPUT_PT}")"

echo ""
echo "=== Bridge: Extract Net Weights ==="
echo "  Input:  ${LATEST_CKPT}"
echo "  Output: ${OUTPUT_PT}"

python scripts/extract_net_weights.py \
    --input "${LATEST_CKPT}" \
    --output "${OUTPUT_PT}"

echo "=== Bridge Complete ==="
echo ""
echo "Stage 3 can now use: ${OUTPUT_PT}"
