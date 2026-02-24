#!/bin/bash
# Stage 3: Self-Forcing training for StableAvatar
#
# GAN + autoregressive rollout training with SyncNet evaluation.
# Requires ode_init.pt from Stage 2 (or set STUDENT_CKPT).
#
# Usage:
#   bash scripts/stableavatar_stage3_sf.sh [OPTIONS] [EXTRA_OPTS...]
#
# Options:
#   --batch_size N        Training batch size per GPU (default: 1)
#   --grad_accum N        Gradient accumulation steps (default: 1)
#   --save_every N        Save checkpoint every N iters (default: 1000)
#   --val_every N         Run validation every N iters (default: 500)
#   --max_iter N          Max training iterations (default: 10000)
#
# Examples:
#   # Default settings
#   bash scripts/stableavatar_stage3_sf.sh
#
#   # Custom training schedule
#   bash scripts/stableavatar_stage3_sf.sh \
#       --batch_size 2 --grad_accum 4 --save_every 500 --val_every 250
#
#   # Pass extra config overrides after --
#   bash scripts/stableavatar_stage3_sf.sh --batch_size 2 -- model.guidance_scale=7.0
#
# Environment variables:
#   NGPUS                   Number of GPUs (auto-detected)
#   WANDB_PROJECT           WandB project name (default: stableavatar)
#   WANDB_API_KEY           WandB API key
#   RUN_NAME                Run name (default: sf_<timestamp>)
#   STUDENT_CKPT            Path to ode_init.pt (default: $CKPT_ROOT_DIR/stableavatar/ode_init.pt)

set -euo pipefail

export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-"FASTGEN_OUTPUT"}
export CKPT_ROOT_DIR=${CKPT_ROOT_DIR:-"${FASTGEN_OUTPUT_ROOT}/MODEL"}

NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}
RUN_NAME=${RUN_NAME:-"sf_$(date +%y%m%d_%H%M)"}

# Defaults
BATCH_SIZE=1
GRAD_ACCUM=1
SAVE_EVERY=1000
VAL_EVERY=500
MAX_ITER=10000
EXTRA_OPTS=()

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --grad_accum)  GRAD_ACCUM="$2";  shift 2 ;;
        --save_every)  SAVE_EVERY="$2";  shift 2 ;;
        --val_every)   VAL_EVERY="$2";   shift 2 ;;
        --max_iter)    MAX_ITER="$2";    shift 2 ;;
        --)            shift; EXTRA_OPTS+=("$@"); break ;;
        *)             EXTRA_OPTS+=("$1"); shift ;;
    esac
done

# Verify ode_init.pt exists
STUDENT_CKPT=${STUDENT_CKPT:-"${CKPT_ROOT_DIR}/stableavatar/ode_init.pt"}
if [ ! -f "${STUDENT_CKPT}" ]; then
    echo "ERROR: Student checkpoint not found at ${STUDENT_CKPT}"
    echo "Run Stage 2 first, or set STUDENT_CKPT to the correct path."
    exit 1
fi

echo "=== Stage 3: Self-Forcing Training ==="
echo "  GPUs:          ${NGPUS}"
echo "  Run name:      ${RUN_NAME}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  Grad accum:    ${GRAD_ACCUM}"
echo "  Max iter:      ${MAX_ITER}"
echo "  Save every:    ${SAVE_EVERY} iters"
echo "  Val every:     ${VAL_EVERY} iters"
echo "  Student ckpt:  ${STUDENT_CKPT}"
echo ""

PYTHONPATH=$(pwd) torchrun \
    --nproc_per_node="${NGPUS}" \
    --standalone \
    train.py \
    --config=fastgen/configs/experiments/StableAvatar/config_sf.py \
    - trainer.fsdp=True \
    log_config.name="${RUN_NAME}" \
    dataloader_train.batch_size="${BATCH_SIZE}" \
    trainer.grad_accum_rounds="${GRAD_ACCUM}" \
    trainer.save_ckpt_iter="${SAVE_EVERY}" \
    trainer.validation_iter="${VAL_EVERY}" \
    trainer.max_iter="${MAX_ITER}" \
    model.pretrained_student_net_path="${STUDENT_CKPT}" \
    "${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"}"

echo "=== Stage 3 Complete ==="
