import numpy as np


def cca(
        comp_tracks,
        ref_tracks,
        labels_ref,
        labels_comp,
        mapped_ref,
        mapped_comp,
):
    """
    Computes the cell cycle accuracy. As described in the paper,
         "An objective comparison of cell-tracking algorithms."
           - Vladimir Ulman et al., Nature methods 2017

    Args:
        comp_tracks: The result tracks.
        ref_tracks: The ground truth tracks.
        labels_ref: The labels of the ground truth masks.
        labels_comp: The labels of the result masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The cell cycle accuracy metric.
    """

    is_parent_ref = np.isin(ref_tracks[:, 0], ref_tracks[:, 3])
    is_child_ref = np.isin(ref_tracks[:, 3], ref_tracks[:, 0])
    valid_ref = np.logical_and(is_parent_ref, is_child_ref)
    track_lengths_ref = np.sum(ref_tracks[:, 2] - ref_tracks[:, 1])
    if np.sum(valid_ref) == 0:
        return None
    track_lengths_ref = track_lengths_ref[valid_ref]

    is_parent_comp = np.isin(comp_tracks[:, 0], comp_tracks[:, 3])
    is_child_comp = np.isin(comp_tracks[:, 3], comp_tracks[:, 0])
    valid_comp = np.logical_and(is_parent_comp, is_child_comp)
    track_lengths_comp = np.sum(comp_tracks[:, 2] - comp_tracks[:, 1])
    if np.sum(valid_comp) == 0:
        return 0
    track_lengths_comp = track_lengths_comp[valid_comp]

    max_track_length = np.max(
        [np.max(track_lengths_ref), np.max(track_lengths_comp)])
    hist_ref = np.arange(0, np.max(max_track_length) + 1)
    for i in track_lengths_ref:
        hist_ref[i] += 1
    hist_ref = hist_ref / np.sum(hist_ref)
    cum_hist_ref = np.cumsum(hist_ref)
    hist_comp = np.arange(0, np.max(max_track_length) + 1)
    for i in track_lengths_comp:
        hist_comp[i] += 1
    hist_comp = hist_comp / np.sum(hist_comp)
    cum_hist_comp = np.cumsum(hist_comp)

    cca = np.max(np.abs(cum_hist_ref - cum_hist_comp))

    return float(cca)
