import numpy as np

from ctc_metrics.utils.representations import assign_comp_to_ref


def bc(
        comp_tracks,
        ref_tracks,
        labels_ref,
        labels_comp,
        mapped_ref,
        mapped_comp,
        i: int,
):
    """
    Computes the branching correctness metric. As described in the paper,
         "An objective comparison of cell-tracking algorithms."
           - Vladimir Ulman et al., Nature methods 2017

    Args:
        comp_tracks: The result tracks.
        ref_tracks: The ground truth tracks.
        labels_ref: The labels of the ground truth masks.
        labels_comp: The labels of the result masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.
        i: The maximal allowed error in frames.

    Returns:
        The branching correctness metric.
    """

    # Extract relevant tracks with children in reference
    parents_ref, counts_ref = np.unique(ref_tracks[:, 3], return_counts=True)
    counts_ref = counts_ref[parents_ref > 0]
    parents_ref = parents_ref[parents_ref > 0]
    ends_with_split_ref = parents_ref[counts_ref > 1]
    t_ref = np.array([ref_tracks[ref_tracks[:, 0] == ref][0, 2]
                      for ref in ends_with_split_ref])
    if len(ends_with_split_ref) == 0:
        return None

    # Extract relevant tracks with children in computed result
    parents_comp, counts_comp = np.unique(comp_tracks[:, 3], return_counts=True)
    counts_comp = counts_comp[parents_comp > 0]
    parents_comp = parents_comp[parents_comp > 0]
    ends_with_split_comp = parents_comp[counts_comp > 1]
    t_comp = np.asarray([comp_tracks[comp_tracks[:, 0] == comp][0, 2]
                         for comp in ends_with_split_comp])
    if len(ends_with_split_comp) == 0:
        return 0

    # # Match ref to comp that satisfy the branching correctness with max gap of i
    # matches = list()
    # for ref, t in zip(ends_with_split_ref, t_ref):
    #     # Find potential matches
    #     potential_matches = np.abs(t_comp - t) <= i
    #     if len(potential_matches) == 0:
    #         continue
    #     ref_children = ref_tracks[ref_tracks[:, 3] == ref][:, 0]
    #     parent_track = track_assignments[ref]
    #     children_tracks = [track_assignments[x] for x in ref_children]
    #     # Evaluate potential matches
    #     for comp, _t in zip(
    #             ends_with_split_comp[potential_matches],
    #             t_comp[potential_matches]
    #     ):
    #         comp_children = comp_tracks[comp_tracks[:, 3] == comp][:, 0]
    #         if len(ref_children) != len(comp_children):
    #             continue
    #         if t <= _t:
    #             parent_match = parent_track[t] == comp
    #             children_matches = [
    #                 x[_t+1] in comp_children for x in children_tracks
    #             ]
    #
    #         else:
    #             parent_match = parent_track[_t] == comp
    #             children_matches = [
    #                 x[t + 1] in comp_children for x in children_tracks
    #             ]
    #         if parent_match and all(children_matches):
    #             matches.append((ref, comp))
    #             break

    matches = list()
    for comp, tc in zip(ends_with_split_comp, t_comp):
        # Find potential matches
        pot_matches = np.abs(t_ref - tc) <= i
        if len(pot_matches) == 0:
            continue
        comp_children = comp_tracks[comp_tracks[:, 3] == comp][:, 0]
        # Evaluate potential matches
        for ref, tr in zip(
                ends_with_split_ref[pot_matches],
                t_ref[pot_matches]
        ):
            ref_children = ref_tracks[ref_tracks[:, 3] == ref][:, 0]
            if len(ref_children) != len(comp_children):
                continue
            t1, t2 = min(tr, tc), max(tr, tc)
            # Compare parents
            mr, mc = mapped_ref[t1], mapped_comp[t1]
            if np.sum(mc == comp) < 1 or np.sum(mr == ref) != 1:
                continue
            ind = np.argwhere(mr == ref).squeeze()
            if mc[ind] != comp:
                continue
            # Compare children
            mr, mc = np.asarray(mapped_ref[t2+1]), np.asarray(mapped_comp[t2+1])
            if not np.all(np.isin(comp_children, mc)):
                continue
            inds = np.isin(mc, comp_children)
            if not np.all(np.isin(mr[inds], ref_children)):
                continue
            matches.append((ref, comp))

    # Calculate BC(i)
    tp = len(matches)
    fp = len(ends_with_split_comp) - tp
    fn = len(ends_with_split_ref) - tp
    precision = tp / max((tp + fp), 0.0001)
    recall = tp / max((tp + fn), 0.0001)
    f1_score = 2 * (precision * recall) / max((precision + recall), 0.0001)
    return f1_score
