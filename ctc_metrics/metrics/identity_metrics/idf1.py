import numpy as np
from scipy.optimize import linear_sum_assignment

from ctc_metrics.utils.representations import track_confusion_matrix

def get_idf1_stats(
        track_intersection,
        costs,
        max_label_ref,
        max_label_comp
):
    # Hungarian Algorithm
    all_gt = np.sum(track_intersection[1:], axis=1)
    all_comp = np.sum(track_intersection[:, 1:], axis=0)

    row_ind, col_ind = linear_sum_assignment(costs)

    IDFN = all_gt.sum()
    IDFP = all_comp.sum()
    for i, j in zip(row_ind, col_ind):
        if i < max_label_ref and j < max_label_comp:
            IDFN -= track_intersection[i + 1, j + 1]
            IDFP -= track_intersection[i + 1, j + 1]

    assert (track_intersection[1:, :].sum() - IDFN ==
            track_intersection[:, 1:].sum() - IDFP)

    IDTP = track_intersection[1:, :].sum() - IDFN

    IDP = IDTP / (IDTP + IDFP)
    IDR = IDTP / (IDTP + IDFN)
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    return {
        "IDF1": IDF1,
        "IDP": IDP,
        "IDR": IDR,
        "IDTP": IDTP,
        "IDFP": IDFP,
        "IDFN": IDFN
    }

def idf1(
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list
):
    """
    Computes the IDF1 metric. As described in the paper,
         "Performance Measures and a Data Set for
          Multi-Target, Multi-Camera Tracking"
           - Ristani et al., ECCV  2016

    Args:
        labels_comp: The labels of the computed masks.
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The idf1 tracks metric.
    """

    max_label_ref = int(np.max(np.concatenate(labels_ref)))
    max_label_comp = int(np.max(np.concatenate(labels_comp)))
    track_intersection = track_confusion_matrix(
        labels_ref, labels_comp, mapped_ref, mapped_comp
    )

    # Assign Tracks
    total_ids = max_label_ref + max_label_comp
    costs = np.ones((total_ids, total_ids)) * np.inf
    subcosts = np.zeros_like(track_intersection[1:, 1:])
    subcosts[track_intersection[1:, 1:] == 0] = np.inf
    costs[:max_label_ref, :max_label_comp] = subcosts
    costs[:max_label_ref, :max_label_comp] -= 2 * track_intersection[1:, 1:]
    costs[:max_label_ref, :max_label_comp] += np.sum(
        track_intersection[1:, :], axis=1)[:, None]
    costs[:max_label_ref, :max_label_comp] += np.sum(
        track_intersection[:, 1:], axis=0)[None, :]
    costs[max_label_ref:, max_label_comp:] = 0

    for i, c in enumerate(np.sum(track_intersection[1:, :], axis=1)):
        costs[i, max_label_comp + i] = c
    for j, c in enumerate(np.sum(track_intersection[:, 1:], axis=0)):
        costs[max_label_ref + j, j] = c

    return get_idf1_stats(
        track_intersection, costs, max_label_ref, max_label_comp
    )
