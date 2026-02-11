#!/bin/bash

# 3-Stage Self-Forcing Pipeline for Cosmos-Predict2.5-2B (T2W)
#
# This script consolidates the full pipeline:
#   Stage 1: Generate ODE pairs (teacher trajectory capture)
#   Stage 2: CausalKD training (causal student on ODE paths via MSE)
#   Stage 3: Self-Forcing training (GAN + AR rollout refinement)
#
# Prerequisites:
#   - Set FASTGEN_OUTPUT_ROOT and CKPT_ROOT_DIR environment variables
#   - Model checkpoint at $CKPT_ROOT_DIR/cosmos_predict2/Cosmos-Predict2.5-2B/...
#   - Prompt file for ODE pair generation
#   - Video data (WDS format) for Self-Forcing stage
#
# Usage:
#   bash scripts/train_cosmos_sf.sh <stage> [EXTRA_OPTS...]
#
# Stages:
#   1 | ode_pairs     - Generate ODE pair data
#   2 | kd_causal     - CausalKD training
#   bridge            - Extract net weights (Stage 2→3)
#   3 | self_forcing   - Self-Forcing training
#   all               - Run all stages sequentially
#
# Examples:
#   # Generate ODE pairs
#   bash scripts/train_cosmos_sf.sh 1 --prompt_file prompts.txt --output_dir ODE_PAIRS/cosmos_t2w
#
#   # CausalKD training
#   bash scripts/train_cosmos_sf.sh 2
#
#   # Extract weights (bridge)
#   bash scripts/train_cosmos_sf.sh bridge --input checkpoints/0010000.pth --output ode_init.pt
#
#   # Self-Forcing training
#   bash scripts/train_cosmos_sf.sh 3

set -euo pipefail

export FASTGEN_OUTPUT_ROOT=${FASTGEN_OUTPUT_ROOT:-"FASTGEN_OUTPUT"}
export CKPT_ROOT_DIR=${CKPT_ROOT_DIR:-"${FASTGEN_OUTPUT_ROOT}/MODEL"}

NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}

STAGE="${1:-help}"
shift || true

# ============================================================================
# Stage 1: Generate ODE Pairs
# ============================================================================
run_ode_pairs() {
    echo "=== Stage 1: Generate ODE Pairs ==="
    echo "GPUs: ${NGPUS}"

    PYTHONPATH=$(pwd) torchrun \
        --nproc_per_node=${NGPUS} \
        --standalone \
        scripts/generate_ode_pairs.py \
        --config=fastgen/configs/experiments/CosmosPredict2/config_dmd2.py \
        --num_steps=35 \
        --guidance_scale=3.0 \
        --fps=24 \
        - trainer.ddp=True \
        "$@"

    echo "=== Stage 1 Complete ==="
}

# ============================================================================
# Stage 2: CausalKD Training
# ============================================================================
run_kd_causal() {
    echo "=== Stage 2: CausalKD Training ==="
    echo "GPUs: ${NGPUS}"

    RUN_NAME="cosmos_kd_causal_$(date +%y%m%d%H%M)"

    PYTHONPATH=$(pwd) torchrun \
        --nproc_per_node=${NGPUS} \
        --standalone \
        train.py \
        --config=fastgen/configs/experiments/CosmosPredict2/config_kd_causal.py \
        - trainer.fsdp=True \
        log_config.name="${RUN_NAME}" \
        "$@"

    echo "=== Stage 2 Complete ==="
}

# ============================================================================
# Bridge: Extract Net Weights (Stage 2 → Stage 3)
# ============================================================================
run_bridge() {
    echo "=== Bridge: Extract Net Weights ==="

    python scripts/extract_net_weights.py "$@"

    echo "=== Bridge Complete ==="
}

# ============================================================================
# Stage 3: Self-Forcing Training
# ============================================================================
run_self_forcing() {
    echo "=== Stage 3: Self-Forcing Training ==="
    echo "GPUs: ${NGPUS}"

    RUN_NAME="cosmos_sf_$(date +%y%m%d%H%M)"

    PYTHONPATH=$(pwd) torchrun \
        --nproc_per_node=${NGPUS} \
        --standalone \
        train.py \
        --config=fastgen/configs/experiments/CosmosPredict2/config_sf.py \
        - trainer.fsdp=True \
        log_config.name="${RUN_NAME}" \
        "$@"

    echo "=== Stage 3 Complete ==="
}

# ============================================================================
# Dispatch
# ============================================================================
case "${STAGE}" in
    1|ode_pairs)
        run_ode_pairs "$@"
        ;;
    2|kd_causal)
        run_kd_causal "$@"
        ;;
    bridge)
        run_bridge "$@"
        ;;
    3|self_forcing)
        run_self_forcing "$@"
        ;;
    all)
        echo "Running all stages sequentially..."
        echo "Note: You must configure paths between stages."
        echo ""
        run_ode_pairs "$@"
        run_kd_causal "$@"
        echo ">> Run 'bridge' stage manually to extract weights before Stage 3"
        ;;
    help|*)
        echo "Usage: bash scripts/train_cosmos_sf.sh <stage> [EXTRA_OPTS...]"
        echo ""
        echo "Stages:"
        echo "  1 | ode_pairs     Generate ODE pair data (Stage 1)"
        echo "  2 | kd_causal     CausalKD training (Stage 2)"
        echo "  bridge            Extract net weights (Stage 2→3)"
        echo "  3 | self_forcing  Self-Forcing training (Stage 3)"
        echo "  all               Run stages 1+2 sequentially"
        echo ""
        echo "Environment variables:"
        echo "  FASTGEN_OUTPUT_ROOT  Output root directory (default: FASTGEN_OUTPUT)"
        echo "  CKPT_ROOT_DIR        Checkpoint root directory"
        echo "  NGPUS                Number of GPUs (auto-detected)"
        ;;
esac
