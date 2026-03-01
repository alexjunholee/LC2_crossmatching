"""Degree of similarity (ψ) computation for LC2 Phase 1.

Computes the FoV overlap between sensor views on the 2D ground plane.

Optimized for pools with many crops per position: groups entries by
position, computes position-pair visibility once, then derives
per-entry ψ via fast angular overlap.

Reference: Lee et al., "(LC)²", RA-L 2023 — Section III.B.1.
"""

import numpy as np
from typing import List, Tuple


def compute_headings(positions: np.ndarray) -> np.ndarray:
    """Estimate heading angles from a trajectory of (x, y) positions.

    Args:
        positions: (N, 2) array of (x, y) coordinates.

    Returns:
        (N,) array of heading angles in radians.
    """
    N = len(positions)
    headings = np.zeros(N)
    if N < 2:
        return headings
    diffs = np.diff(positions, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms > 1e-6
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings[:-1] = np.where(valid, angles, 0.0)
    # Forward-fill zeros from previous valid heading
    for i in range(1, N - 1):
        if not valid[i]:
            headings[i] = headings[i - 1]
    headings[-1] = headings[-2]
    return headings


def _angular_overlap_deg(h_i: float, fov_i: float, h_j: float, fov_j: float) -> float:
    """Compute angular overlap IoU between two FoV sectors (in degrees).

    Fast O(1) computation for co-located or nearby sensors.

    Returns:
        IoU ∈ [0, 1].
    """
    dh = (h_i - h_j) % 360.0
    if dh > 180.0:
        dh -= 360.0

    half_i = fov_i / 2.0
    half_j = fov_j / 2.0

    overlap = max(0.0, min(dh + half_i, half_j) - max(dh - half_i, -half_j))
    union = fov_i + fov_j - overlap

    if union <= 0:
        return 0.0
    return overlap / union


def build_psi_pairs(
    positions: np.ndarray,
    headings: np.ndarray,
    fovs_deg: np.ndarray,
    max_ranges: np.ndarray,
    max_dist: float = 50.0,
) -> List[Tuple[int, int, float]]:
    """Compute pairwise ψ values — returns sparse list of (i, j, ψ) tuples.

    Strategy:
        1. Group entries by unique position using spatial grid hashing.
        2. For each group pair within max_dist:
           a. If same position: ψ = angular_overlap(heading_i, heading_j).
           b. If different positions: distance-decayed angular overlap.
        3. Skip pairs beyond max_dist entirely.

    This avoids building a dense N×N matrix (which would be ~400MB for N=10K).

    Args:
        positions: (N, 2) sensor positions.
        headings: (N,) heading angles (radians).
        fovs_deg: (N,) horizontal FoVs in degrees.
        max_ranges: (N,) max effective ranges.
        max_dist: Only compute ψ for pairs within this distance.

    Returns:
        List of (i, j, ψ) tuples where ψ > 0.01.
    """
    N = len(positions)

    # ── Group entries by unique position via spatial grid hash ──
    # Grid cell size = 0.5m → entries within same cell are co-located
    grid_cell = 0.5
    pos_groups: List[List[int]] = []
    group_positions_list: List[np.ndarray] = []
    grid_to_group = {}  # (gx, gy) → group_id

    for i in range(N):
        gx = int(np.floor(positions[i, 0] / grid_cell))
        gy = int(np.floor(positions[i, 1] / grid_cell))

        assigned = False
        # Check this cell and neighboring cells
        for dx in range(-1, 2):
            if assigned:
                break
            for dy in range(-1, 2):
                key = (gx + dx, gy + dy)
                if key in grid_to_group:
                    g = grid_to_group[key]
                    if np.linalg.norm(positions[i] - group_positions_list[g]) < grid_cell:
                        pos_groups[g].append(i)
                        assigned = True
                        break

        if not assigned:
            g = len(pos_groups)
            pos_groups.append([i])
            group_positions_list.append(positions[i].copy())
            grid_to_group[(gx, gy)] = g

    n_groups = len(pos_groups)
    group_positions = np.array(group_positions_list)
    print(f"  ψ computation: {N} entries → {n_groups} unique positions")

    # ── Pairwise group distances (vectorized) ──
    g_diff = group_positions[:, None, :] - group_positions[None, :, :]
    g_dists = np.sqrt((g_diff ** 2).sum(axis=2))

    # Convert headings to degrees
    headings_deg = np.degrees(headings)

    # ── Compute ψ for nearby group pairs ──
    psi_entries: List[Tuple[int, int, float]] = []

    for gi in range(n_groups):
        members_i = pos_groups[gi]
        for gj in range(gi, n_groups):
            if g_dists[gi, gj] > max_dist:
                continue

            members_j = pos_groups[gj]

            if gi == gj:
                # Same position group: purely angular overlap
                for a_idx, a in enumerate(members_i):
                    for b in members_i[a_idx + 1:]:
                        psi_val = _angular_overlap_deg(
                            headings_deg[a], fovs_deg[a],
                            headings_deg[b], fovs_deg[b],
                        )
                        if psi_val > 0.01:
                            psi_entries.append((a, b, psi_val))
            else:
                # Different positions: distance-decayed angular overlap
                dist = g_dists[gi, gj]
                max_r = min(max_ranges[members_i[0]], max_ranges[members_j[0]])
                if max_r <= 0:
                    continue
                dist_factor = max(0.0, 1.0 - dist / max_r)
                if dist_factor < 0.01:
                    continue

                # Bearing between groups
                dxy = group_positions[gj] - group_positions[gi]
                bearing_ij = np.degrees(np.arctan2(dxy[1], dxy[0]))
                bearing_ji = bearing_ij + 180.0

                for a in members_i:
                    da = abs(((headings_deg[a] - bearing_ij) + 180) % 360 - 180)
                    if da > fovs_deg[a] / 2.0 + 10:
                        continue

                    for b in members_j:
                        db = abs(((headings_deg[b] - bearing_ji) + 180) % 360 - 180)
                        if db > fovs_deg[b] / 2.0 + 10:
                            continue

                        ang_overlap = _angular_overlap_deg(
                            headings_deg[a], fovs_deg[a],
                            headings_deg[b], fovs_deg[b],
                        )
                        psi_val = ang_overlap * dist_factor
                        if psi_val > 0.01:
                            psi_entries.append((a, b, psi_val))

    print(f"  ψ pairs: {len(psi_entries)} non-zero out of {N*(N-1)//2} total")
    return psi_entries
