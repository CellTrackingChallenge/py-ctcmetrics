import numpy as np

from ctc_metrics.metrics.biological.bc import get_ids_that_ends_with_split


def is_valid_track(
        tracks: np.ndarray,
):
    """
    Extracts the boolean indices of valid tracks that start and end with a
        cell split.

    Args:
        tracks: The tracks to check.

    Returns:
        The boolean indices of valid tracks that start and end with a cell
            split.
    """
    ends_with_split = get_ids_that_ends_with_split(tracks)
    is_parent = np.isin(tracks[:, 0], ends_with_split)
    is_child = np.isin(tracks[:, 3], ends_with_split)
    valid = np.logical_and(is_parent, is_child)
    return valid


def cca(
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray
):
    """
    Computes the cell cycle accuracy. As described in the paper,
         "An objective comparison of cell-tracking algorithms."
           - Vladimir Ulman et al., Nature methods 2017

    Args:
        comp_tracks: The result tracks. A (n,4) numpy ndarray with columns:
            - label
            - birth frame
            - end frame
            - parent
        ref_tracks: The ground truth tracks. A (n,4) numpy ndarray with columns:
            - label
            - birth frame
            - end frame
            - parent

    Returns:
        The cell cycle accuracy metric.
    """

    # Extract relevant tracks with parents and children in reference
    valid_ref = is_valid_track(ref_tracks)
    if np.sum(valid_ref) == 0:
        return None
    track_lengths_ref = ref_tracks[valid_ref, 2] - ref_tracks[valid_ref, 1]
    # Extract relevant tracks with parents and children in computed result
    valid_comp = is_valid_track(comp_tracks)
    if np.sum(valid_comp) == 0:
        return 0
    track_lengths_comp = comp_tracks[valid_comp, 2] - comp_tracks[valid_comp, 1]
    # Calculate CCA
    max_track_length = np.max(
        [np.max(track_lengths_ref), np.max(track_lengths_comp)])
    hist_ref = np.zeros(np.max(max_track_length) + 1)
    for i in track_lengths_ref:
        hist_ref[i] += 1
    hist_ref = hist_ref / np.sum(hist_ref)
    cum_hist_ref = np.cumsum(hist_ref)
    hist_comp = np.zeros(np.max(max_track_length) + 1)
    for i in track_lengths_comp:
        hist_comp[i] += 1
    hist_comp = hist_comp / np.sum(hist_comp)
    cum_hist_comp = np.cumsum(hist_comp)
    return float(1 - np.max(np.abs(cum_hist_ref - cum_hist_comp)))
