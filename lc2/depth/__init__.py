"""Pluggable depth estimation backends for LC2.

This subpackage provides a unified factory interface over multiple
monocular / stereo depth estimation models.  All heavy dependencies
(depth-anything-3, DUSt3R, DINOv3 hub weights, ...) are imported
lazily so that installing one backend does **not** require installing
the others.

Quick start
-----------
>>> from lc2.depth import get_depth_estimator, list_depth_estimators
>>> print(list_depth_estimators())
['da3', 'da3-small', 'da3-large', 'da3-giant', 'dinov3', 'dust3r']
>>> estimator = get_depth_estimator("da3")
>>> depth = estimator.estimate("image.png")
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

from lc2.depth.base import BaseDepthEstimator

__all__ = [
    "BaseDepthEstimator",
    "get_depth_estimator",
    "list_depth_estimators",
]

# ---------------------------------------------------------------------------
# Lazy estimator registry
#
# Each entry maps a user-facing short name to (module_path, class_name).
# The module is imported only when the estimator is actually requested.
# ---------------------------------------------------------------------------
_ESTIMATOR_CLASSES: Dict[str, Tuple[str, str]] = {
    # Depth Anything 3 variants
    "da3":       ("lc2.depth.depth_anything", "DA3DepthEstimator"),
    "da3-small": ("lc2.depth.depth_anything", "DA3DepthEstimator"),
    "da3-large": ("lc2.depth.depth_anything", "DA3DepthEstimator"),
    "da3-giant": ("lc2.depth.depth_anything", "DA3DepthEstimator"),
    # DINOv3 DPT
    "dinov3":    ("lc2.depth.dinov3_depth",   "DINOv3DepthEstimator"),
    # DUSt3R
    "dust3r":    ("lc2.depth.dust3r_depth",   "DUSt3RDepthEstimator"),
}

# Which kwargs should be forwarded as `variant` instead of being passed
# directly to the constructor.  For DA3 and DINOv3 the short name *is*
# the variant selector.
_VARIANT_MAP: Dict[str, str] = {
    "da3":       "da3-large",
    "da3-small": "da3-small",
    "da3-large": "da3-large",
    "da3-giant": "da3-giant",
    "dinov3":    "dinov3_vitl14",
}


def list_depth_estimators() -> List[str]:
    """Return the list of registered estimator short names."""
    return sorted(_ESTIMATOR_CLASSES.keys())


def get_depth_estimator(
    name: str,
    device: Optional[str] = None,
    **kwargs: Any,
) -> BaseDepthEstimator:
    """Instantiate a depth estimator by short name.

    Args:
        name: One of the keys from :func:`list_depth_estimators`
            (e.g. ``"da3"``, ``"da3-small"``, ``"dinov3"``, ``"dust3r"``).
        device: CUDA device string.  Auto-detected if *None*.
        **kwargs: Extra keyword arguments forwarded to the estimator
            constructor (e.g. ``image_size`` for DUSt3R).

    Returns:
        An initialised :class:`BaseDepthEstimator` instance.

    Raises:
        KeyError: If *name* is not a registered estimator.
    """
    if name not in _ESTIMATOR_CLASSES:
        raise KeyError(
            f"Unknown depth estimator '{name}'.  "
            f"Available: {list_depth_estimators()}"
        )

    module_path, class_name = _ESTIMATOR_CLASSES[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Inject `variant` if the backend uses one and the caller didn't
    # supply it explicitly.
    if name in _VARIANT_MAP and "variant" not in kwargs:
        kwargs["variant"] = _VARIANT_MAP[name]

    return cls(device=device, **kwargs)
