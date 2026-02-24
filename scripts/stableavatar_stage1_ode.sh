#!/bin/bash
# Stage 1: Generate ODE pairs for StableAvatar CausalKD training
#
# Runs the bidirectional teacher through a 40-step ODE solver and captures
# intermediate denoising states. Writes latent.pth + path.pth per sample.
# Skips samples that already have both files.
#
# Usage:
#   bash scripts/stableavatar_stage1_ode.sh [EXTRA_ARGS...]
#
# Examples:
#   bash scripts/stableavatar_stage1_ode.sh
#   bash scripts/stableavatar_stage1_ode.sh --num_samples 100
#
# Environment variables:
#   STABLEAVATAR_DATA_DIR   Input data directory (default: /home/work/stableavatar_data/v2v_training_data)
#   STABLEAVATAR_CKPT       Teacher checkpoint (default: /home/work/.local/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt)
#   ODE_OUTPUT_DIR           Output directory for ODE pairs (default: same as data dir)
#   NUM_STEPS                ODE solver steps (default: 40)
#   GUIDANCE_SCALE           CFG scale (default: 5.0)

set -euo pipefail

DATA_DIR=${STABLEAVATAR_DATA_DIR:-"/home/work/stableavatar_data/v2v_training_data"}
CKPT=${STABLEAVATAR_CKPT:-"/home/work/.local/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"}
OUTPUT_DIR=${ODE_OUTPUT_DIR:-"${DATA_DIR}"}
NUM_STEPS=${NUM_STEPS:-40}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}

echo "=== Stage 1: Generate ODE Pairs ==="
echo "  Data dir:       ${DATA_DIR}"
echo "  Checkpoint:     ${CKPT}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  ODE steps:      ${NUM_STEPS}"
echo "  Guidance scale: ${GUIDANCE_SCALE}"
echo ""

PYTHONPATH=$(pwd) python scripts/generate_stableavatar_ode_pairs.py \
    --data_dir "${DATA_DIR}" \
    --checkpoint "${CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_steps "${NUM_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    "$@"

echo "=== Stage 1 Complete ==="
