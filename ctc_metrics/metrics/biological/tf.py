import numpy as np

from ctc_metrics.utils.representations import assign_comp_to_ref


def tf(
    comp_tracks,
    ref_tracks,
    labels_ref,
    labels_comp,
    mapped_ref,
    mapped_comp,
):
    """
    Computes the track fractions metric. As described in the paper,
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
        The track fractions metric.
    """

    track_assignments = assign_comp_to_ref(
        labels_ref, labels_comp, mapped_ref, mapped_comp
    )

    tfs = list()
    for k, v in track_assignments.items():
        uniques, counts = np.unique(v, return_counts=True)
        tfs.append(np.max(counts) / len(v))

    tf = np.mean(tfs)
    return float(tf)
