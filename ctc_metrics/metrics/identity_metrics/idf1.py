import numpy as np
from scipy.optimize import linear_sum_assignment


def idf1(
        ref_tracks: np.ndarray,
        comp_tracks: np.ndarray,
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

    # Gather association data
    track_intersection = np.zeros((max_label_ref + 1, max_label_comp + 1))
    total_ref = np.zeros(max_label_ref + 1)
    total_comp = np.zeros(max_label_comp + 1)
    for ref, comp, m_ref, m_comp in zip(
            labels_ref, labels_comp, mapped_ref, mapped_comp):
        # Fill track intersection matrix
        ref = np.asarray(ref)
        comp = np.asarray(comp)
        m_ref = np.asarray(m_ref)
        m_comp = np.asarray(m_comp)
        _, counts = np.unique(m_comp, return_counts=True)
        double_associations = np.sum(counts[counts > 1] - 1)
        if len(m_ref) > 0:
            track_intersection[m_ref, m_comp] += 1
        fna = ref[np.isin(ref, m_ref, invert=True)]
        track_intersection[fna, 0] += 1
        fpa = comp[np.isin(comp, m_comp, invert=True)]
        track_intersection[0, fpa] += 1
        total_comp[comp] += 1
        total_ref[ref] += 1
        assert len(ref) == len(m_ref) + len(fna), (len(ref), len(m_ref), len(fna), double_associations)
        assert len(comp) + double_associations == len(m_comp) + len(fpa), (len(comp), len(m_comp), len(fpa), double_associations)

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

    res = {
        "IDF1": IDF1,
        "IDP": IDP,
        "IDR": IDR,
        "IDTP": IDTP,
        "IDFP": IDFP,
        "IDFN": IDFN
    }

    return res

