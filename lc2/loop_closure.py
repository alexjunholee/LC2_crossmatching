"""Loop closure filtering via pose graph optimization (GTSAM).

Filters false-positive loop closures from cross-modal place recognition
by constructing a factor graph with odometry and loop closure factors,
then filtering by the information matrix of each loop edge.

Reference: Lee et al., "(LC)²", RA-L 2023 — Section III.D.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import gtsam
    from gtsam import Pose2, noiseModel

    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False


def filter_loop_closures(
    query_poses: np.ndarray,
    db_poses: np.ndarray,
    matches: np.ndarray,
    match_distances: np.ndarray,
    odom_cov: float = 1e-2,
    loop_cov: float = 1e4,
    dist_threshold: float = 0.1,
    info_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter false-positive loop closures using pose graph optimization.

    Constructs a factor graph from odometry (query trajectory) and
    raw loop closures (cross-modal matches). Solves the MAP estimate
    and filters loops by the L2-norm of diagonal information.

    Args:
        query_poses: (N, 3) query poses [x, y, theta].
        db_poses: (M, 3) database poses [x, y, theta].
        matches: (N,) index of best match in DB for each query.
        match_distances: (N,) descriptor distances of each match.
        odom_cov: Odometry noise covariance (diagonal, tight).
        loop_cov: Loop closure noise covariance (diagonal, loose).
        dist_threshold: Maximum descriptor distance for a valid loop.
        info_threshold: Minimum information norm to keep a loop.

    Returns:
        valid_mask: (N,) boolean mask of accepted loop closures.
        info_norms: (N,) L2-norm of diagonal information per loop.
    """
    if not HAS_GTSAM:
        print("[loop_closure] GTSAM not installed. Falling back to distance-only filtering.")
        valid = match_distances < dist_threshold
        return valid, match_distances

    N = len(query_poses)

    # Build factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    odom_noise = noiseModel.Diagonal.Sigmas(
        np.array([odom_cov, odom_cov, odom_cov * 0.1])
    )
    loop_noise = noiseModel.Diagonal.Sigmas(
        np.array([loop_cov, loop_cov, loop_cov * 0.1])
    )

    # Add prior on first pose
    prior_noise = noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.001]))
    graph.add(gtsam.PriorFactorPose2(0, _to_pose2(query_poses[0]), prior_noise))
    initial.insert(0, _to_pose2(query_poses[0]))

    # Add odometry factors
    for i in range(1, N):
        initial.insert(i, _to_pose2(query_poses[i]))
        odom = _to_pose2(query_poses[i - 1]).between(_to_pose2(query_poses[i]))
        graph.add(gtsam.BetweenFactorPose2(i - 1, i, odom, odom_noise))

    # Add loop closure factors (only for matches with low descriptor distance)
    db_node_offset = N  # DB nodes start at index N
    loop_indices = []
    for i in range(N):
        if match_distances[i] < dist_threshold:
            db_idx = int(matches[i])
            db_key = db_node_offset + db_idx

            if not initial.exists(db_key):
                initial.insert(db_key, _to_pose2(db_poses[db_idx]))

            # Loop factor: query_i ↔ db_match
            relative = _to_pose2(query_poses[i]).between(
                _to_pose2(db_poses[db_idx])
            )
            graph.add(gtsam.BetweenFactorPose2(i, db_key, relative, loop_noise))
            loop_indices.append(i)

    if len(loop_indices) == 0:
        return np.zeros(N, dtype=bool), np.full(N, np.inf)

    # Optimize with Levenberg-Marquardt
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    # Compute marginal covariances for loop closure filtering
    try:
        marginals = gtsam.Marginals(graph, result)
    except RuntimeError:
        # If marginals fail, fall back to distance-only filtering
        valid = match_distances < dist_threshold
        return valid, match_distances

    # Filter by information norm
    valid_mask = np.zeros(N, dtype=bool)
    info_norms = np.full(N, 0.0)

    for i in loop_indices:
        db_idx = int(matches[i])
        db_key = db_node_offset + db_idx

        try:
            # Joint marginal covariance between query node and DB node
            keys = gtsam.KeyVector()
            keys.push_back(i)
            keys.push_back(db_key)
            joint_cov = marginals.jointMarginalCovariance(keys).fullMatrix()

            # Information = inverse of marginal covariance
            info_matrix = np.linalg.inv(joint_cov + 1e-10 * np.eye(joint_cov.shape[0]))
            diag_info = np.diag(info_matrix)
            info_norm = np.linalg.norm(diag_info)

            info_norms[i] = info_norm
            valid_mask[i] = info_norm > info_threshold
        except (RuntimeError, np.linalg.LinAlgError):
            pass

    return valid_mask, info_norms


def _to_pose2(pose: np.ndarray) -> "gtsam.Pose2":
    """Convert [x, y, theta] array to gtsam.Pose2."""
    return gtsam.Pose2(float(pose[0]), float(pose[1]),
                       float(pose[2]) if len(pose) > 2 else 0.0)
