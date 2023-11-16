import numpy as np
from multiprocessing import Pool, cpu_count

from ctc_metrics.utils.representations import cluster_full_tracks


def find_match(
    k1,
    k2_candidates,
    start_gt,
    end_gt,
    start_res,
    end_res,
    cluster_gt,
    cluster_res,
    labels_gt,
    labels_res,
    mapped_gt,
    mapped_res,
):
    """
    Helper function for the track fraction metric.
    Finds the maximal overlap between a track and a set of tracks.

    Args:
        k1: The key of the track to match.
        k2_candidates: The keys of the tracks to match against.
        start_gt: The start frames of the ground truth tracks.
        end_gt: The end frames of the ground truth tracks.
        start_res: The start frames of the result tracks.
        end_res: The end frames of the result tracks.
        cluster_gt: The clustered ground truth tracks.
        cluster_res: The clustered result tracks.
        matches: The matches between the result and ground truth frames.

    Returns:
        A tuple of k1 and The maximal overlap.

    """
    matched_track = 0
    start1, end1 = start_gt[k1], end_gt[k1]
    track1 = cluster_gt[k1]
    length1 = end1 - start1 + 1
    for k2 in k2_candidates:
        if matched_track > 0.5:
            break
        max_correct, current_correct = 0, 0
        track2 = cluster_res[k2]
        start2, end2 = start_res[k2], end_res[k2]
        start = max(start1, start2)
        end = min(end1, end2)
        for frame in range(start, end + 1):
            if max_correct < current_correct:
                max_correct = current_correct
            if max_correct > (current_correct + end - frame + 1):
                break
            labels1, labels2, mapped1, mapped2 = labels_gt[frame], \
                labels_res[frame], mapped_gt[frame], mapped_res[frame]
            id1, id2 = np.isin(track1, labels1), np.isin(track2, labels2)
            if np.sum(id1) != np.sum(id2):
                current_correct = 0
                continue
            if np.sum(id1) == 0:
                current_correct += 1
                continue
            id1 = track1[int(np.argwhere(id1))]
            id2 = track2[int(np.argwhere(id2))]
            mapped_id1 = np.argwhere(mapped1 == id1)
            mapped_id2 = np.argwhere(mapped2 == id2)
            if mapped_id1.size == 0 or mapped_id2.size == 0:
                current_correct = 0
                continue
            if int(mapped_id1) != int(mapped_id2):
                current_correct = 0
            else:
                current_correct += 1
                continue
        if max_correct < current_correct:
            max_correct = current_correct
        max_fraction = max_correct / length1
        if max_fraction > matched_track:
            matched_track = max_fraction

    return k1, matched_track


def tf(
    res_tracks,
    gt_tracks,
    labels_gt,
    labels_res,
    mapped_gt,
    mapped_res,
    multiprocessing=True
):
    """
    Computes the track fractions metric.

    # Checked against test datasets -> OK

    Args:
        res_tracks: The result tracks.
        gt_tracks: The ground truth tracks.
        existing_ground_truth_frames: The frames in the ground truth.
        matches: The matches between the result and ground truth tracks.

    Returns:
        The track fractions metric.
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
    # Find tracks that have potential matches
    potential_matched_tracks = dict()
    for k1, v1 in clustered_gt.items():
        potential_matched_tracks[k1] = list()
        start1, end1 = start_frames_gt[k1], end_frames_gt[k1]
        for k2, v2 in clustered_res.items():
            start2, end2 = start_frames_res[k2], end_frames_res[k2]
            if start1 > end2 or start2 > end1:
                continue
            potential_matched_tracks[k1].append(k2)

    # Find matches

    args = [(
        k, v, start_frames_gt, end_frames_gt, start_frames_res, end_frames_res,
        clustered_gt, clustered_res, labels_gt, labels_res, mapped_gt,
        mapped_res
    ) for k, v in potential_matched_tracks.items()]
    if multiprocessing:
        with Pool(cpu_count()) as p:
            matched_tracks = p.starmap(
                find_match,
                args
            )
    else:
        matched_tracks = [find_match(*arg) for arg in args]
    matched_tracks = dict(matched_tracks)

    tf = np.sum(list(matched_tracks.values())) / num_gt_tracks
    return tf
