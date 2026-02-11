"""Extract network weights from a DDP/FSDP checkpoint (Stage 2→3 bridge).

Extracts the `net` (or specified key) state_dict from a training checkpoint
and saves it as a flat .pt file suitable for `pretrained_student_net_path`.

Examples:

    # Extract net weights from CausalKD checkpoint
    python scripts/extract_net_weights.py \\
        --input checkpoints/0010000.pth \\
        --output ode_init.pt

    # Extract with custom key prefix
    python scripts/extract_net_weights.py \\
        --input checkpoints/0010000.pth \\
        --output ode_init.pt \\
        --key net
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import fastgen.utils.logging_utils as logger


def extract_weights(input_path: str, output_path: str, key: str = "net") -> None:
    """Extract network weights from a checkpoint file.

    Args:
        input_path: Path to the input checkpoint (.pth)
        output_path: Path to save the extracted weights (.pt)
        key: Key prefix to extract from the checkpoint state_dict
    """
    logger.info(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Extract keys with the specified prefix
    prefix = f"{key}."
    extracted = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            extracted[k[len(prefix) :]] = v

    if not extracted:
        # Try without prefix (maybe the state_dict is already the net's)
        logger.warning(f"No keys found with prefix '{prefix}', saving full state_dict")
        extracted = state_dict

    logger.info(f"Extracted {len(extracted)} parameters")

    # Log some statistics
    total_params = sum(v.numel() for v in extracted.values())
    logger.info(f"Total parameters: {total_params:,}")

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, output_path)
    logger.success(f"Saved extracted weights to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract network weights from a training checkpoint")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the extracted weights (.pt)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="net",
        help="Key prefix to extract from the checkpoint (default: 'net')",
    )

    args = parser.parse_args()
    extract_weights(args.input, args.output, args.key)
