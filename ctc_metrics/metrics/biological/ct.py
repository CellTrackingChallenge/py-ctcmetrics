import numpy as np

from ctc_metrics.utils.representations import assign_comp_to_ref


def ct(
        comp_tracks,
        ref_tracks,
        labels_ref,
        labels_comp,
        mapped_ref,
        mapped_comp,
):
    """
    Computes the complete tracks metric. As described in the paper,
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
        The complete tracks metric.
    """

    track_assignments = assign_comp_to_ref(
        labels_ref, labels_comp, mapped_ref, mapped_comp
    )

    tfs = list()
    for k, v in track_assignments.items():
        uniques, counts = np.unique(v, return_counts=True)
        tfs.append(np.max(counts) / len(v))

    T_rc = np.sum(np.asarray(tfs) == 1)

    T_r = len(ref_tracks)
    T_c = len(comp_tracks)

    ct = float(2 * T_rc / (T_c + T_r))
    return float(ct)


if __name__ == "__main__":
    print(np.unique([1,2,3,4,5], return_counts=True))
    uniques, counts = np.unique([1,2,3,4,5], return_counts=True)
    for i,c in zip(uniques, counts):
        print(i,c)
