import numpy as np

from ctc_metrics.utils.representations import assign_comp_to_ref


def calculate_fractions_fo_computed_tracks(
        ref_tracks: np.ndarray,
        labels_ref: list,
        mapped_ref: list,
        mapped_comp: list,
):
    """
    Returns a dictionary with the fractions of overlap for each computed track.
    Each entry in the dictionary is a dictionary with the computed track id as
    key and a new dictionary as value. In the sub-dictionary, the reference
    track id is the key and the fraction of overlap is the value.

    Args:
        ref_tracks: The ground truth tracks.
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        A dictionary with the fractions of overlap for each computed track.
    """
    track_assignments = assign_comp_to_ref(labels_ref, mapped_ref, mapped_comp)
    comp_fractions = {}
    for k, v in track_assignments.items():
        start_ref = ref_tracks[ref_tracks[:, 0] == k][0, 1]
        end_ref = ref_tracks[ref_tracks[:, 0] == k][0, 2]
        length = int(end_ref - start_ref + 1)
        array = v[start_ref:end_ref + 1]
        assigned_labels = np.unique(array)
        if len(assigned_labels) == 1 and assigned_labels[0] == 0:
            continue
        assignments = 0
        last_i = 0
        for i in array:
            if last_i != i or i == 0:
                assignments = 0
            if i > 0:
                assignments += 1
                if i not in comp_fractions:
                    comp_fractions[i] = {}
                comp_fractions[i][k] = max(
                    comp_fractions[i].get(k, 0), assignments / length)
            last_i = i
    return comp_fractions


def tf(
        ref_tracks: np.ndarray,
        labels_ref: list,
        mapped_ref: list,
        mapped_comp: list
):
    """
    Computes the track fractions metric. As described in the paper,
         "An objective comparison of cell-tracking algorithms."
           - Vladimir Ulman et al., Nature methods 2017

    Args:
        ref_tracks: The ground truth tracks.
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The track fractions metric.
    """
    comp_fractions = calculate_fractions_fo_computed_tracks(
        ref_tracks, labels_ref, mapped_ref, mapped_comp)
    # Calculate the track fractions with respect to the reference tracks
    tfs = {k: 0 for k in ref_tracks[:, 0]}
    for k, v in sorted(comp_fractions.items()):
        for k2, v2 in sorted(v.items()):
            if tfs[k2] == 1:
                continue
            if v2 > tfs[k2]:
                tfs[k2] = v2
            # ###
            # The next lines do not make sense, because it causes instabilities.
            # Exactly same results with different order of labels can lead to
            # different metrics. They are kept, because they are in the original
            # codebase for the CTC. Remove the break to solve the issue.
            # Note: Does not make real differences in most cases.
            # ###
            if v2 == 1:
                break
    # Filter out undetected tracks. They should not be counted.
    tfs = [x for k, x in tfs.items() if x > 0]
    if len(tfs) == 0:
        return 0
    return np.mean(np.asarray(tfs))
