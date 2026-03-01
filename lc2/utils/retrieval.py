"""FAISS-based retrieval and place recognition evaluation metrics.

Provides functions for building FAISS indices, performing top-K retrieval,
and computing standard VPR metrics (Recall@K, precision-recall curves).
"""

import numpy as np
import faiss
from typing import Dict, List, Optional, Set, Tuple


def build_index(descriptors: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS L2 (Euclidean) flat index from descriptor matrix.

    Args:
        descriptors: Database descriptors of shape ``(N, D)``, float32.

    Returns:
        A FAISS IndexFlatL2 populated with the given descriptors.
    """
    descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
    dim = descriptors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(descriptors)
    return index


def retrieve(
    queries: np.ndarray,
    index: faiss.IndexFlatL2,
    top_k: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve top-K nearest database entries for each query.

    Args:
        queries: Query descriptors of shape ``(M, D)``, float32.
        index: FAISS index built from database descriptors.
        top_k: Number of nearest neighbors to retrieve.

    Returns:
        Tuple of:
        - distances: L2 distances, shape ``(M, top_k)``.
        - indices: Database indices, shape ``(M, top_k)``.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    distances, indices = index.search(queries, top_k)
    return distances, indices


def compute_gt_positives(
    query_positions: np.ndarray,
    db_positions: np.ndarray,
    threshold: float = 25.0,
) -> List[Set[int]]:
    """Compute ground-truth positive matches based on spatial distance.

    For each query, finds all database entries within ``threshold`` meters.

    Args:
        query_positions: Query positions, shape ``(M, 2)`` or ``(M, 3)``.
            Only the first 2 columns (x, y) are used for distance.
        db_positions: Database positions, shape ``(N, 2)`` or ``(N, 3)``.
        threshold: Distance threshold in meters for a positive match.

    Returns:
        List of length M, where each element is a set of positive database
        indices for the corresponding query.
    """
    # Use only xy for distance (ground plane)
    q_xy = query_positions[:, :2].astype(np.float64)
    d_xy = db_positions[:, :2].astype(np.float64)

    # Pairwise distance matrix: (M, N)
    # dist²(q, d) = ||q||² + ||d||² - 2 * q · d
    q_sq = np.sum(q_xy ** 2, axis=1, keepdims=True)  # (M, 1)
    d_sq = np.sum(d_xy ** 2, axis=1, keepdims=True)  # (N, 1)
    dist_sq = q_sq + d_sq.T - 2.0 * q_xy @ d_xy.T  # (M, N)
    dist_sq = np.maximum(dist_sq, 0.0)  # numerical safety
    dist = np.sqrt(dist_sq)

    threshold_f = float(threshold)
    positives: List[Set[int]] = []
    for i in range(len(q_xy)):
        pos_indices = set(np.where(dist[i] < threshold_f)[0].tolist())
        positives.append(pos_indices)

    return positives


def compute_recall(
    predictions: np.ndarray,
    gt_positives: List[Set[int]],
    ks: List[int] = [1, 5, 10, 15, 20],
) -> Dict[int, float]:
    """Compute Recall@K for top-K retrieval predictions.

    A query is considered correctly recalled at rank K if any of its top-K
    retrieved database entries is in the ground-truth positive set.

    Args:
        predictions: Retrieved database indices, shape ``(M, max_K)``.
        gt_positives: Ground-truth positive sets, one per query.
        ks: List of K values to evaluate.

    Returns:
        Dict mapping K → recall value in [0, 1].
    """
    num_queries = len(gt_positives)
    if num_queries == 0:
        return {k: 0.0 for k in ks}

    recalls: Dict[int, float] = {}
    for k in ks:
        correct = 0
        evaluated = 0
        for i in range(num_queries):
            if len(gt_positives[i]) == 0:
                continue  # skip queries with no ground-truth positives
            evaluated += 1
            top_k_preds = set(predictions[i, :k].tolist())
            if top_k_preds & gt_positives[i]:
                correct += 1
        recalls[k] = correct / max(evaluated, 1)

    return recalls


def compute_pr_curve(
    distances: np.ndarray,
    is_correct: np.ndarray,
    num_thresholds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision-recall curve from retrieval distances.

    Sweeps a distance threshold from min to max, computing precision and
    recall at each threshold.

    Args:
        distances: L2 distances for top-1 retrievals, shape ``(M,)``.
        is_correct: Boolean array indicating if top-1 match is correct, shape ``(M,)``.
        num_thresholds: Number of threshold points for the curve.

    Returns:
        Tuple of (precision, recall) arrays, each of shape ``(num_thresholds,)``.
    """
    thresholds = np.linspace(distances.min(), distances.max(), num_thresholds)

    precision = np.zeros(num_thresholds)
    recall = np.zeros(num_thresholds)

    total_correct = is_correct.sum()
    if total_correct == 0:
        return precision, recall

    for i, t in enumerate(thresholds):
        # At threshold t, we "accept" all matches with distance <= t
        accepted = distances <= t
        accepted_correct = (accepted & is_correct).sum()
        accepted_total = accepted.sum()

        precision[i] = accepted_correct / max(accepted_total, 1)
        recall[i] = accepted_correct / total_correct

    return precision, recall


def evaluate_retrieval(
    query_descriptors: np.ndarray,
    db_descriptors: np.ndarray,
    query_positions: np.ndarray,
    db_positions: np.ndarray,
    pos_threshold: float = 25.0,
    ks: List[int] = [1, 5, 10],
    top_k: int = 25,
) -> Dict:
    """End-to-end retrieval evaluation pipeline.

    Builds FAISS index, retrieves top-K matches, computes ground truth
    positives, and returns all metrics.

    Args:
        query_descriptors: Query descriptors ``(M, D)``.
        db_descriptors: Database descriptors ``(N, D)``.
        query_positions: Query positions ``(M, 2+)``.
        db_positions: Database positions ``(N, 2+)``.
        pos_threshold: Distance threshold for positive match (meters).
        ks: Recall@K values to compute.
        top_k: Number of nearest neighbors to retrieve.

    Returns:
        Dict with keys:
        - ``"recall@K"`` for each K in ks
        - ``"predictions"``: ``(M, top_k)`` index array
        - ``"distances"``: ``(M, top_k)`` distance array
        - ``"gt_positives"``: list of positive sets
        - ``"num_queries"``: total queries evaluated
        - ``"num_db"``: database size
    """
    # Build index and retrieve
    index = build_index(db_descriptors)
    distances, predictions = retrieve(query_descriptors, index, top_k=top_k)

    # Ground truth
    gt_positives = compute_gt_positives(query_positions, db_positions, pos_threshold)

    # Recall@K
    recalls = compute_recall(predictions, gt_positives, ks=ks)

    # Top-1 correctness for PR curve
    top1_correct = np.array([
        predictions[i, 0] in gt_positives[i]
        for i in range(len(gt_positives))
    ], dtype=bool)

    result = {
        "predictions": predictions,
        "distances": distances,
        "gt_positives": gt_positives,
        "top1_correct": top1_correct,
        "num_queries": len(query_positions),
        "num_db": len(db_positions),
    }
    for k, v in recalls.items():
        result[f"recall@{k}"] = v

    return result
