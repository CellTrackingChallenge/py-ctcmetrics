import numpy as np


def cca(
        res_tracks,
        gt_tracks,
):
    """
    Computes the cell cycle accuracy metric.

    # Checked against test datasets -> OK

    Args:
        res_tracks: The result tracks.
        gt_tracks: The ground truth tracks.
        existing_ground_truth_frames: The frames in the ground truth.
        matches: The matches between the result and ground truth tracks.

    Returns:
        The cell cycle accuracy metric.
    """
    # Calculate mitosis distribution in ground_truth
    gt_track_lengths = list()
    parent_label, num_children = np.unique(gt_tracks[:, 3], return_counts=True)
    num_children[parent_label == 0] = 0
    has_two_children = parent_label[num_children > 1]
    has_one_child = parent_label[num_children == 1]
    has_sibling = gt_tracks[np.isin(gt_tracks[:, 3], has_two_children), 0]
    if has_sibling.size == 0:
        return None
    for label in has_sibling:
        total_length = 0
        current_label = label
        while True:
            idx = np.where(gt_tracks[:, 0] == current_label)[0][0]
            total_length += 1 + gt_tracks[idx, 2] - gt_tracks[idx, 1]
            if current_label in has_two_children:
                gt_track_lengths.append(total_length)
                break
            elif current_label in has_one_child:
                idx = np.where(gt_tracks[:, 3] == current_label)[0][0]
                current_label = gt_tracks[idx, 0]
            else:
                break
    if len(gt_track_lengths) == 0:
        return None
    # Calculate mitosis distribution in results
    res_track_lengths = list()
    parent_label, num_children = np.unique(res_tracks[:, 3], return_counts=True)
    num_children[parent_label == 0] = 0
    has_two_children = parent_label[num_children > 1]
    has_one_child = parent_label[num_children == 1]
    has_sibling = res_tracks[np.isin(res_tracks[:, 3], has_two_children), 0]
    if has_sibling.size == 0:
        return 0
    for label in has_sibling:
        total_length = 0
        current_label = label
        while True:
            idx = np.where(res_tracks[:, 0] == current_label)[0][0]
            total_length += 1 + res_tracks[idx, 2] - res_tracks[idx, 1]
            if current_label in has_two_children:
                res_track_lengths.append(total_length)
                break
            elif current_label in has_one_child:
                idx = np.where(res_tracks[:, 3] == current_label)[0][0]
                current_label = res_tracks[idx, 0]
            else:
                break
    if len(res_track_lengths) == 0:
        return 0
    # Compare distributions
    max_len = max([max(res_track_lengths),max(gt_track_lengths)])
    res_dist, gt_dist = np.zeros(max_len+1), np.zeros(max_len+1)
    for i in res_track_lengths:
        res_dist[i-1] += 1
    for i in gt_track_lengths:
        gt_dist[i-1] += 1
    res_dist_cdf = np.cumsum(res_dist)/ np.sum(res_dist)
    gt_dist_cdf = np.cumsum(gt_dist) / np.sum(gt_dist)
    diff = np.abs(res_dist_cdf - gt_dist_cdf)
    cca = 1 - np.max(diff)
    return float(cca)
