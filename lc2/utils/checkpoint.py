"""Checkpoint loading utilities for LC2.

Handles the original LC2 checkpoint format (DataParallel wrapped, combined
encoder+NetVLAD state_dict) and provides clean extraction of component weights.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


def load_lc2_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Load an LC2 checkpoint and clean up DataParallel key prefixes.

    The original LC2 codebase wraps models in ``nn.DataParallel``, which
    prepends ``module.`` to all state_dict keys. This function strips that
    prefix and returns a clean state_dict.

    Args:
        path: Path to the ``.pth.tar`` checkpoint file.
        device: Device to map tensors to during loading.

    Returns:
        Tuple of (state_dict, metadata).
        - state_dict: Cleaned parameter dict with ``module.`` prefix stripped.
        - metadata: Dict with any additional checkpoint info (epoch, best_metric, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle different checkpoint structures
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            raw_state = checkpoint["model_state_dict"]
        else:
            # Assume the dict itself is the state_dict
            raw_state = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # Strip DataParallel "module." prefixes.
    # The original LC2 uses nested DataParallel:
    #   outer DataParallel → module.encoder.module.encoder_d.0.weight
    #   inner DataParallel → encoder.module.encoder_d.0.weight
    # We strip the leading "module." AND any interior ".module." → "."
    state_dict = {}
    for key, value in raw_state.items():
        clean_key = key
        # Strip leading "module." (outer DataParallel)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        # Strip interior ".module." (inner DataParallel on sub-modules)
        clean_key = clean_key.replace(".module.", ".")
        state_dict[clean_key] = value

    # Extract metadata
    meta = {}
    for meta_key in ("epoch", "best_score", "best_metric", "optimizer", "config"):
        if isinstance(checkpoint, dict) and meta_key in checkpoint:
            meta[meta_key] = checkpoint[meta_key]

    return state_dict, meta


def extract_encoder_netvlad_state(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a combined state_dict into encoder and NetVLAD components.

    Args:
        state_dict: Combined state_dict from ``load_lc2_checkpoint``.

    Returns:
        Tuple of (encoder_state, netvlad_state).
    """
    encoder_state = {}
    netvlad_state = {}

    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state[key] = value
        elif key.startswith("pool."):
            netvlad_state[key] = value
        else:
            # Try to classify by known parameter names
            if any(vgg_key in key for vgg_key in ("features.", "encoder_d.", "encoder_r.")):
                encoder_state[key] = value
            elif any(vlad_key in key for vlad_key in ("centroids", "conv.weight", "conv.bias")):
                netvlad_state[key] = value

    return encoder_state, netvlad_state
