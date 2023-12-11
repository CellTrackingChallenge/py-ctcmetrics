import numpy as np


def get_ids_that_ends_with_split(
        tracks: np.ndarray
):
    """
    Extracts the ids of tracks that end with a cell split.

    Args:
        tracks: The tracks to check.

    Returns:
        The ids of tracks that end with a cell split.
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
        comp: int,
        ref: int,
        mapped_ref: list,
        mapped_comp: list,
        ref_children: np.ndarray,
        comp_children: np.ndarray,
        tr: int,
        tc: int
):
    """
    Checks if the reference and the computed track match.

    Args:
        comp: The computed track id.
        ref: The reference track id.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.
        ref_children: The children ids of the reference track.
        comp_children: The children ids of the computed track.
        tr: The reference track end.
        tc: The computed track end.

    Returns:
        True if the reference and the computed track match, False otherwise.
    """
    # Check if the number of children is the same
    if len(ref_children) != len(comp_children):
        return False
    # Compare parents
    t1, t2 = min(tr, tc), max(tr, tc)
    mr, mc = mapped_ref[t1], mapped_comp[t1]
    if np.sum(mc == comp) < 1 or np.sum(mr == ref) != 1:
        return False
    ind = np.argwhere(mr == ref).squeeze()
    if mc[ind] != comp:
        return False
    # Compare children
    mr, mc = np.asarray(mapped_ref[t2 + 1]), np.asarray(mapped_comp[t2 + 1])
    if not np.all(np.isin(comp_children, mc)):
        return False
    if not np.all(np.isin(mr[np.isin(mc, comp_children)], ref_children)):
        return False
    return True


def bc(
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray,
        mapped_ref: list,
        mapped_comp: list,
        i: int
):
    """
    Computes the branching correctness metric. As described in the paper,
         "An objective comparison of cell-tracking algorithms."
           - Vladimir Ulman et al., Nature methods 2017

    Args:
        comp_tracks: The result tracks.
        ref_tracks: The ground truth tracks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.
        i: The maximal allowed error in frames.

    Returns:
        The branching correctness metric.
    """

    # Extract relevant tracks with children in reference
    ends_with_split_ref = get_ids_that_ends_with_split(ref_tracks)
    t_ref = np.array([ref_tracks[ref_tracks[:, 0] == ref][0, 2]
                      for ref in ends_with_split_ref])
    if len(ends_with_split_ref) == 0:
        return None
    # Extract relevant tracks with children in computed result
    ends_with_split_comp = get_ids_that_ends_with_split(comp_tracks)
    t_comp = np.asarray([comp_tracks[comp_tracks[:, 0] == comp][0, 2]
                         for comp in ends_with_split_comp])
    if len(ends_with_split_comp) == 0:
        return 0
    # Find all matches between reference and computed branching events (mitosis)
    matches = []
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
            if is_matching(
                    comp, ref, mapped_ref, mapped_comp, ref_children,
                    comp_children, tr, tc
            ):
                matches.append((ref, comp))
    # Calculate BC(i)
    return calculate_f1_score(
        len(matches),
        len(ends_with_split_comp) - len(matches),
        len(ends_with_split_ref) - len(matches))
