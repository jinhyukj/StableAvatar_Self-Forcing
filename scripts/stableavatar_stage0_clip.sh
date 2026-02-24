#!/bin/bash
# Stage 0: Precompute CLIP features for StableAvatar data
#
# One-time preprocessing step. Computes clip_fea.pt [257, 1280] per sample
# from reference frames. Skips samples that already have clip_fea.pt.
#
# Usage:
#   bash scripts/stableavatar_stage0_clip.sh [DATA_DIR ...]
#
# Examples:
#   # Default: process training + validation data
#   bash scripts/stableavatar_stage0_clip.sh
#
#   # Custom directories
#   bash scripts/stableavatar_stage0_clip.sh /path/to/data1 /path/to/data2

set -euo pipefail

CLIP_MODEL=${CLIP_MODEL:-"openai/clip-vit-large-patch14-336"}
VAE_MODEL=${VAE_MODEL:-"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}

# Default directories to process
DEFAULT_DIRS=(
    "/home/work/stableavatar_data/v2v_training_data"
    "/home/work/stableavatar_data/v2v_validation_data/recon"
    "/home/work/stableavatar_data/v2v_validation_data/mixed"
)

if [ $# -gt 0 ]; then
    DIRS=("$@")
else
    DIRS=("${DEFAULT_DIRS[@]}")
fi

for DATA_DIR in "${DIRS[@]}"; do
    echo "=== Precomputing CLIP features: ${DATA_DIR} ==="

    PYTHONPATH=$(pwd) python scripts/precompute_stableavatar_clip.py \
        --data_dir "${DATA_DIR}" \
        --clip_model "${CLIP_MODEL}" \
        --vae_model "${VAE_MODEL}"

    echo "=== Done: ${DATA_DIR} ==="
    echo ""
done

echo "Stage 0 complete."
