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

    correct_tracks = 0
    for k, v in track_assignments.items():
        start_ref = ref_tracks[ref_tracks[:, 0] == k][0, 1]
        end_ref = ref_tracks[ref_tracks[:, 0] == k][0, 2]
        assigned_labels = np.unique(v[~np.isnan(v)])
        if len(assigned_labels) > 1:
            continue
        if assigned_labels[0] == 0:
            continue
        assignee = assigned_labels[0]
        start_comp = comp_tracks[comp_tracks[:, 0] == assignee][0, 1]
        end_comp = comp_tracks[comp_tracks[:, 0] == assignee][0, 2]
        if start_ref == start_comp and end_ref == end_comp:
            correct_tracks += 1

    T_rc = correct_tracks
    T_r = len(ref_tracks)
    T_c = len(comp_tracks)
    ct = float(2 * T_rc / (T_c + T_r))
    return float(ct)


if __name__ == "__main__":
    print(np.unique([1,2,3,4,5], return_counts=True))
    uniques, counts = np.unique([1,2,3,4,5], return_counts=True)
    for i,c in zip(uniques, counts):
        print(i,c)
