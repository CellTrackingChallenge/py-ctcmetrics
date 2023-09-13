import numpy as np


def bc(
        res_tracks,
        gt_tracks,
        labels_gt,
        labels_res,
        mapped_gt,
        mapped_res,
        i: int,
):
    """
    Computes the branching correctness metric.

    # Checked against test datasets -> OK

    Args:
        res_tracks: The result tracks.
        gt_tracks: The ground truth tracks.
        existing_ground_truth_frames: The frames in the ground truth.
        matches: The matches between the result and ground truth tracks.
        i: The maximal frame distance.

    Returns:
        The branching correctness metric.
    """
    # Find branches
    parent_label, num_children = np.unique(gt_tracks[:, 3], return_counts=True)
    num_children[parent_label == 0] = 0
    branches_gt = parent_label[num_children > 1]
    if len(branches_gt) == 0:
        return None
    parent_label, num_children = np.unique(res_tracks[:, 3], return_counts=True)
    num_children[parent_label == 0] = 0
    branches_res = parent_label[num_children > 1]
    # Find matches
    _gt, _res = np.copy(branches_gt), np.copy(branches_res)
    matched_branches = list()
    for b1 in _gt:
        end1 = gt_tracks[gt_tracks[:, 0] == b1, 2][0]
        for j, b2 in enumerate(_res):
            end2 = res_tracks[res_tracks[:, 0] == b2, 2][0]
            if abs(end1 - end2) >= (2 * i + 1):
                continue
            frame = min(end1, end2)
            labels1, labels2, mapped1, mapped2 = labels_gt[frame], \
                labels_res[frame], mapped_gt[frame], mapped_res[frame]
            if b1 in mapped1 and b2 in mapped2:
                mapped_b1 = int(np.argwhere(mapped1 == b1))
                mapped_b2 = int(np.argwhere(mapped2 == b2))
                if mapped_b1 != mapped_b2:
                    continue
                matched_branches.append((b1, b2))
                _res = np.delete(_res, j)
                break
    bci = 2 * len(matched_branches) / (len(branches_gt) + len(branches_res))
    return bci
