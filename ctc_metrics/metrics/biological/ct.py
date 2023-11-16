import numpy as np

from ctc_metrics.utils.representations import cluster_full_tracks


def ct(
        res_tracks,
        gt_tracks,
        labels_gt,
        labels_res,
        mapped_gt,
        mapped_res,
):
    """
    Computes the complete tracks metric.

    # Checked against test datasets -> OK

    Args:
        res_tracks: The result tracks.
        gt_tracks: The ground truth tracks.
        existing_ground_truth_frames: The frames in the ground truth.
        matches: The matches between the result and ground truth tracks.

    Returns:
        The complete tracks metric.
    """
    clustered_res = cluster_full_tracks(res_tracks)
    clustered_gt = cluster_full_tracks(gt_tracks)
    start_frames_gt = {
        k: gt_tracks[gt_tracks[:, 0] == v[0], 1][0] for k, v in
        clustered_gt.items()
    }
    start_frames_res = {
        k: res_tracks[res_tracks[:, 0] == v[0], 1][0] for k, v in
        clustered_res.items()
    }
    end_frames_gt = {
        k: gt_tracks[gt_tracks[:, 0] == v[-1], 2][0] for k, v in
        clustered_gt.items()
    }
    end_frames_res = {
        k: res_tracks[res_tracks[:, 0] == v[-1], 2][0] for k, v in
        clustered_res.items()
    }
    num_gt_tracks = len(clustered_gt)
    num_res_tracks = len(clustered_res)
    # Find tracks that have potential matches
    potential_matched_tracks = list()
    for k, v in clustered_gt.items():
        start, end = start_frames_gt[k], end_frames_gt[k]
        for k2, v2 in clustered_res.items():
            start2, end2 = start_frames_res[k2], end_frames_res[k2]
            if start == start2 and end == end2:
                potential_matched_tracks.append((k, k2))
    # Find matches
    matched_tracks = dict()
    for k1, k2 in potential_matched_tracks:
        track1, track2 = clustered_gt[k1], clustered_res[k2]
        if len(track1) != len(track2):
            continue
        is_match = True
        for id1, id2 in zip(track1, track2):
            start1, end1 = gt_tracks[gt_tracks[:, 0] == id1, 1:3][0]
            start2, end2 = res_tracks[res_tracks[:, 0] == id2, 1:3][0]
            if start1 != start2 or end1 != end2:
                is_match = False
                break
            for frame in range(start1, end1 + 1):
                labels1, labels2, mapped1, mapped2 = labels_gt[frame],\
                    labels_res[frame], mapped_gt[frame], mapped_res[frame]
                if id1 not in mapped1 or id2 not in mapped2:
                    is_match = False
                    break
                if np.argwhere(mapped1 == id1) != np.argwhere(mapped2 == id2):
                    is_match = False
                    break
        if is_match:
            assert k1 not in matched_tracks
            matched_tracks[k1] = k2
    ct = 2 * len(matched_tracks) / (num_gt_tracks + num_res_tracks)
    return ct
