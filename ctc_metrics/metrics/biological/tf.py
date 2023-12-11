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

    comp_fractions = dict()
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
                    comp_fractions[i] = dict()
                comp_fractions[i][k] = max(
                    comp_fractions[i].get(k, 0), assignments / length)
            last_i = i

    tfs = {k: 0 for k in ref_tracks[:, 0]}
    for k in sorted(comp_fractions.keys()):
        v = comp_fractions[k]
        for k2 in sorted(v.keys()):
            v2 = v[k2]
            if tfs[k2] == 1:
                continue
            if v2 > tfs[k2]:
                tfs[k2] = v2
            """
            The next lines do not make sense, because it causes instabilities.
            Exactly the same results with different order of labels can lead to
            different metrics. They are kept, because they are in the original
            codebase for the CTC. Remove the break to solve the issue. 
            Note: Does not make real differences in most cases. 
            """
            if v2 == 1:
                break

    # Filter out elements that are completely undetected
    # print("All tracks before:", len(tfs))

    tfs = [x for k, x in tfs.items() if x > 0]
    # print("All detected tracks:", len(tfs))
    #
    # print("Complete:", len([x for x in tfs if x == 1]))
    # print("Partly:", len(tfs) - len([x for x in tfs if x == 1]))

    tf = np.mean(np.asarray(tfs))
    return float(tf)
