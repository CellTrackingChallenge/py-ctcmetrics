import numpy as np

from ctc_metrics.utils.representations import track_confusion_matrix


def cluster_clique(
        tracks: np.ndarray
):
    """
    Clusters the tracks into cliques. A clique is a set of tracks that are
    connected by parent-child relationships.

    Args:
        tracks: The result tracks.

    Returns:
        The cliques. A dictionary with the track id as key and the clique as
        value.
    """
    # Cluster ref cliques
    track_id = np.unique(tracks[:, 0]).astype(int)
    track_parents = np.asarray(
        [tracks[tracks[:, 0] == x][0, 3] for x in track_id]).astype(int)
    cliques = {x: [] for x in track_id}

    # forward
    for track in track_id:
        clique = np.asarray([track])
        while True:
            is_child = np.isin(track_parents, clique)
            is_not_in_clique = np.isin(track_id, clique, invert=True)
            new_children = track_id[is_child & is_not_in_clique]
            if len(new_children) == 0:
                break
            clique = np.concatenate((clique, new_children))
        cliques[track] = clique

    # backward
    for track in track_id:
        clique = np.zeros(0)
        for ancestor in track_id:
            if track in cliques[ancestor]:
                clique = np.concatenate((clique, [ancestor]))
        cliques[track] = np.concatenate((cliques[track], clique))

    for track in track_id:
        cliques[track] = np.unique(cliques[track]).astype(int)

    return cliques


def chota(
        ref_tracks: np.ndarray,
        comp_tracks: np.ndarray,
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list
):
    """


    Args:
        ref_tracks: The ground truth tracks. A (n,4) numpy ndarray with columns:
            - label
            - birth frame
            - end frame
            - parent
        comp_tracks: The computed tracks. A (n,4) numpy ndarray with columns:
            - label
            - birth frame
            - end frame
            - parent
        labels_comp: The labels of the computed masks. A list of length equal
            to the number of frames. Each element is a list with the labels of
            the computed masks in the respective frame.
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
        The CHOTA tracks metric.
    """
    # Gather association data
    cliques_ref = cluster_clique(ref_tracks)
    cliques_comp = cluster_clique(comp_tracks)
    track_intersection = track_confusion_matrix(
        labels_ref, labels_comp, mapped_ref, mapped_comp)
    # Calculate Association scores
    chota_score = 0
    for i in range(1, int(np.max(np.concatenate(labels_ref))) + 1):
        for j in range(1, int(np.max(np.concatenate(labels_comp))) + 1):
            if track_intersection[i, j] > 0:
                cliques_ref_i = cliques_ref[i]
                cliques_comp_j = cliques_comp[j]
                roi1 = np.zeros_like(track_intersection, dtype=bool)
                roi2 = np.zeros_like(track_intersection, dtype=bool)
                roi1[cliques_ref_i, :] = True
                roi2[:, cliques_comp_j] = True
                # Calculate the HOTA score
                tpa = np.sum(track_intersection[roi1 & roi2])
                fna = np.sum(track_intersection[cliques_ref_i, :]) - tpa
                fpa = np.sum(track_intersection[:, cliques_comp_j]) - tpa
                # Reweight and add to hota score
                num_pixels = track_intersection[i, j]
                a_corr = tpa / (tpa + fna + fpa)
                chota_score += num_pixels * a_corr
    # Calculate the finalCHOTA score
    tp = track_intersection[1:, 1:].sum()
    fp = track_intersection[0, 1:].sum()
    fn = track_intersection[1:, 0].sum()
    chota_score = np.sqrt(chota_score / (tp + fp + fn))
    return {"CHOTA": chota_score}


