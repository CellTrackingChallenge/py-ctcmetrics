import numpy as np


def get_ids_that_ends_with_split(
        tracks: np.ndarray
):
    """
    Extracts the ids of tracks that end with a cell split.

    Args:
        tracks: The tracks to check. A numpy nd array with columns:
            - label
            - birth frame
            - end frame
            - parent

    Returns:
        The ids of tracks that end with a cell split stored in a numpy.ndarray.
    """
    parents, counts = np.unique(tracks[:, 3], return_counts=True)
    counts = counts[parents > 0]
    parents = parents[parents > 0]
    ends_with_split = parents[counts > 1]
    return ends_with_split


def calculate_f1_score(
        tp: int,
        fp: int,
        fn: int
):
    """
    Calculates the f1 score.

    Args:
        tp: The number of true positives.
        fp: The number of false positives.
        fn: The number of false negatives.

    Returns:
        The f1 score.
    """
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1_score = 2 * (precision * recall) / max((precision + recall), 0.0001)
    return f1_score


def is_matching(
        id_comp: int,
        id_ref: int,
        mapped_ref: list,
        mapped_comp: list,
        ref_children: np.ndarray,
        comp_children: np.ndarray,
        t_parent_end_ref: int,
        t_parent_end_comp: int,
        t_child_start_ref: list,
        t_child_start_comp: list,
        max_i: int,
):
    """
    Checks if the reference and the computed track match.

    Args:
        id_comp: The computed track id.
        id_ref: The reference track id.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.
        ref_children: The children ids of the reference track.
        comp_children: The children ids of the computed track.
        t_parent_end_ref: The frame of the reference track end.
        t_parent_end_comp: The frame of the computed track end.
        t_child_start_ref: The frame of the reference track start.
        t_child_start_comp: The frame of the computed track start.
        max_i: The maximal time gap between ends of the reference and
               computed mother tracks, and beginnings of daughter tracks.
    Returns:
        True if the reference and the computed track match, False otherwise.
    """
    # Check if the number of children is the same
    if len(ref_children) != len(comp_children):
        return False
    # Compare parents, for temporal distance and then for spatial overlap
    if abs(t_parent_end_ref - t_parent_end_comp) > max_i:
        return False
    t_last_common = min(t_parent_end_ref, t_parent_end_comp)
    mr, mc = mapped_ref[t_last_common], mapped_comp[t_last_common]
    if np.sum(mc == id_comp) < 1 or np.sum(mr == id_ref) != 1:
        return False
    ind = np.argwhere(mr == id_ref).squeeze()
    if mc[ind] != id_comp:
        return False
    # Compare children
    #  Iterate over all GT ids and check if the first detection is matched to the correct reference children
    #  See discussion here https://github.com/CellTrackingChallenge/py-ctcmetrics/issues/22
    matched_children = []
    for i, t_ref in zip(ref_children, t_child_start_ref):
        for j, t_comp in zip(comp_children, t_child_start_comp):
            # Check if start frames of the daughters are close enough <= i_max
            temporal_error = abs(t_ref - t_comp)
            if temporal_error > max_i:
                break
            # Verify if children are overlapping spatially
            t_max = max(t_ref, t_comp)
            if i in mapped_ref[t_max] and j in mapped_comp[t_max]:
                ind = mapped_ref[t_max].index(i)
                if mapped_comp[t_max][ind] == j:
                    # There is a match!
                    if j not in matched_children:
                        matched_children.append(j)
                    break

    if len(matched_children) != len(ref_children):
        return False

    return True


def raw_division_metrics(
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray,
        mapped_ref: list,
        mapped_comp: list,
        i: int
):
    """
    Computes number of true positives, false positives, and false negatives for divisions.

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
        i: The maximal allowed temporal error (offset) in frames.

    Returns:
        Tuple of true positives, false positives, and false negatives.
    """
    # Extract relevant tracks with children in reference
    ends_with_split_ref = get_ids_that_ends_with_split(ref_tracks)
    t_ref = np.array([ref_tracks[ref_tracks[:, 0] == ref][0, 2]
                      for ref in ends_with_split_ref])

    # Extract relevant tracks with children in computed result
    ends_with_split_comp = get_ids_that_ends_with_split(comp_tracks)
    t_comp = np.asarray([comp_tracks[comp_tracks[:, 0] == comp][0, 2]
                         for comp in ends_with_split_comp])

    # If there are no divisions in the reference
    if len(ends_with_split_ref) == 0:
        return (0, len(ends_with_split_comp), 0)

    # If there are no divisions in the computed result
    if len(ends_with_split_comp) == 0:
        return (0, 0, len(ends_with_split_ref))

    # Find all matches between reference and computed branching events (mitosis)
    matches = []
    for comp, t_parent_end_start in zip(ends_with_split_comp, t_comp):
        # Find potential matches
        pot_matches = np.abs(t_ref - t_parent_end_start) <= i
        if len(pot_matches) == 0:
            continue
        comp_children = comp_tracks[comp_tracks[:, 3] == comp][:, 0]
        t_child_start_comp = []
        for j in comp_children:
            t = comp_tracks[comp_tracks[:, 0] == j][0, 1]
            t_child_start_comp.append(t)
        # Evaluate potential matches
        for ref, t_parent_end_ref in zip(
                ends_with_split_ref[pot_matches],
                t_ref[pot_matches]
        ):
            ref_children = ref_tracks[ref_tracks[:, 3] == ref][:, 0]
            t_child_start_ref = []
            for j in ref_children:
                t = ref_tracks[ref_tracks[:, 0] == j][0, 1]
                t_child_start_ref.append(t)
            if is_matching(
                    comp, ref,
                    mapped_ref, mapped_comp,
                    ref_children, comp_children,
                    t_parent_end_ref, t_parent_end_start,
                    t_child_start_ref,
                    t_child_start_comp,
                    i
            ):
                matches.append((ref, comp))
    return (len(matches), len(ends_with_split_comp) - len(matches), len(ends_with_split_ref) - len(matches))


def bc(
        tp: int,
        fp: int,
        fn: int
):
    """
    Computes the branching correctness metric. As described in the paper,
         "An objective comparison of cell-tracking algorithms."
           - Vladimir Ulman et al., Nature methods 2017

    Args:
        tp: The number of true positives.
        fp: The number of false positives.
        fn: The number of false negatives.

    Returns:
        The branching correctness metric.
    """
    # Return None if no split is existing in the reference data
    if (tp + fn) == 0:
        return None

    # Calculate BC(i)
    return calculate_f1_score(tp, fp, fn)
