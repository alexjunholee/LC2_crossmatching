"""Backward-compatible shim for ``lc2.depth_estimation.DepthEstimator``.

The depth estimation logic has been moved to :mod:`lc2.depth`.  This
module re-exports :class:`~lc2.depth.depth_anything.DA3DepthEstimator`
under its original name so that existing code like::

    from lc2.depth_estimation import DepthEstimator

continues to work without modification.

For new code, prefer the pluggable factory interface::

    from lc2.depth import get_depth_estimator
    estimator = get_depth_estimator("da3")
"""

from lc2.depth.depth_anything import DA3DepthEstimator as DepthEstimator

__all__ = ["DepthEstimator"]
