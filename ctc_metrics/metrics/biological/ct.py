import numpy as np

from ctc_metrics.utils.representations import assign_comp_to_ref


def ct(
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray,
        labels_ref: list,
        mapped_ref: list,
        mapped_comp: list
):
    """
    Computes the complete tracks metric. As described in the paper,
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
        The complete tracks metric.
    """
    track_assignments = assign_comp_to_ref(labels_ref, mapped_ref, mapped_comp)
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
    return float(2 * T_rc / (T_c + T_r))

