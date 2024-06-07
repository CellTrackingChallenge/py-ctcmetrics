import numpy as np
from scipy.optimize import linear_sum_assignment

from ctc_metrics.utils.representations import track_confusion_matrix


def get_idf1_stats(
        track_intersection,
        costs,
        max_label_ref,
        max_label_comp
):
    """
    Computes the IDF1 stats.

    Args:
        track_intersection: The track intersection matrix.
        costs: The assignment costs matrix.
        max_label_ref: The maximum label of the reference masks.
        max_label_comp: The maximum label of the computed masks.

    Returns:
        The IDF1 stats.
    """
    # Get optimal assignment
    row_ind, col_ind = linear_sum_assignment(costs)
    # Accumulate trackwise number of matches
    all_gt = np.sum(track_intersection[1:], axis=1)
    all_comp = np.sum(track_intersection[:, 1:], axis=0)
    # Compute IDFN, IDFP
    IDFN = all_gt.sum()
    IDFP = all_comp.sum()
    for i, j in zip(row_ind, col_ind):
        if i < max_label_ref and j < max_label_comp:
            IDFN -= track_intersection[i + 1, j + 1]
            IDFP -= track_intersection[i + 1, j + 1]

    assert (track_intersection[1:, :].sum() - IDFN ==
            track_intersection[:, 1:].sum() - IDFP)
    # Compute IDTP
    IDTP = track_intersection[1:, :].sum() - IDFN
    # Compute IDF1, IDP, IDR
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
        labels_comp: The labels of the computed masks. A list of length equal
            to the number of frames. Each element is a list with the labels of
            the computed masks in the respective frame.
        labels_ref: The labels of the ground truth masks. A list of length
            equal to the number of frames. Each element is a list with the
            labels of the ground truth masks in the respective frame.
        mapped_ref: The matched labels of the ground truth masks. A list of
            length equal to the number of frames. Each element is a list with
            the matched labels of the ground truth masks in the respective
            frame. The elements are in the same order as the corresponding
            elements in mapped_comp.
        mapped_comp: The matched labels of the result masks. A list of length
            equal to the number of frames. Each element is a list with the
            matched labels of the result masks in the respective frame. The
            elements are in the same order as the corresponding elements in
            mapped_ref.

    Returns:
        The idf1 tracks metric.
    """
    # Calculate the cop-ref track intersections
    max_label_ref = int(np.max(np.concatenate(labels_ref)))
    max_label_comp = int(np.max(np.concatenate(labels_comp)))
    track_intersection = track_confusion_matrix(
        labels_ref, labels_comp, mapped_ref, mapped_comp
    )
    # Create assignment costs matrx with default of inf costs + Line for the
    #  dummy node (no association)
    total_ids = max_label_ref + max_label_comp
    costs = np.ones((total_ids, total_ids)) * np.inf
    # Fill in the costs of intersection between  tracks
    subcosts = np.zeros_like(track_intersection[1:, 1:])
    subcosts[track_intersection[1:, 1:] == 0] = np.inf
    costs[:max_label_ref, :max_label_comp] = subcosts
    costs[:max_label_ref, :max_label_comp] -= 2 * track_intersection[1:, 1:]
    costs[:max_label_ref, :max_label_comp] += np.sum(
        track_intersection[1:, :], axis=1)[:, None]
    costs[:max_label_ref, :max_label_comp] += np.sum(
        track_intersection[:, 1:], axis=0)[None, :]
    # Set the assignment costs to the dummy nodes
    costs[max_label_ref:, max_label_comp:] = 0
    for i, c in enumerate(np.sum(track_intersection[1:, :], axis=1)):
        costs[i, max_label_comp + i] = c
    for j, c in enumerate(np.sum(track_intersection[:, 1:], axis=0)):
        costs[max_label_ref + j, j] = c

    return get_idf1_stats(
        track_intersection, costs, max_label_ref, max_label_comp
    )
