import numpy as np
import tifffile as tiff
from sklearn.metrics import confusion_matrix


def match(gt_mask_path, res_mask_path):
    """
    Matches the labels of two masks.

    # Checked against test datasets -> OK

    args:
        gt_mask_path: Path to the ground truth mask.
        res_mask_path: Path to the result mask.

    returns:
        A tuple of four numpy arrays. The first array contains the existing
        labels in the ground truth mask. The second array contains the
        existing labels in the result mask. The third array contains the
        matched labels in the ground truth mask. The fourth array contains
        the corresponding matched labels in the result mask.
    """
    map1 = tiff.imread(gt_mask_path)
    map2 = tiff.imread(res_mask_path)
    # Calculate the IoU matrix of the two masks
    labels1, labels2 = np.unique(map1), np.unique(map2)
    offset = int(np.max(labels1) + 1)
    map2 += offset
    cm = confusion_matrix(map1.flatten(), map2.flatten())
    cm[0, :] = 0
    norm1 = cm / (np.sum(cm, axis=1, keepdims=True) + 0.0000001)
    norm2 = cm / (np.sum(cm, axis=0, keepdims=True) + 0.0000001)
    iogt = norm1 * norm2
    iogt[len(labels1):] = 0
    iogt[:, :len(labels1)+1] = 0
    iogt = iogt[1:len(labels1), 1 + len(labels1):]
    # Suppress matches all non-maxima with respect to the ground truth
    iogt[iogt < 0.5] = 0
    iogt[iogt >= 0.5] = 1
    # Remove double assigned detections
    col_sum = np.sum(iogt, axis=0, keepdims=False)
    iogt[:, col_sum > 1] = 0
    # Create Mapping
    rows, cols = np.nonzero(iogt)
    labels1 = labels1[1:]
    labels2 = labels2[1:]
    mapped1 = labels1[rows].tolist()
    mapped2 = labels2[cols].tolist()
    assert np.unique(mapped1).size == len(mapped1), f"a {gt_mask_path} {len(labels1), len(mapped1), iogt.shape, cm.shape, iogt}"
    assert np.unique(mapped2).size == len(mapped2), f"b {gt_mask_path} {len(labels1), len(mapped1), iogt.shape, cm.shape, labels1,labels2, mapped1, mapped2, rows, cols}"
    labels1 = labels1.tolist()
    labels2 = labels2.tolist()
    return labels1, labels2, mapped1, mapped2


def cluster_full_tracks(tracks):
    """ Clusters tracks list into clusters that belong to the same real track.

    # Checked against test datasets -> OK

    Args:
        tracks: A numpy array of shape (N, 4) where N is the number of tracks.
            Each row represents a track and contains four numbers:
                Label Begin End Parent

    Returns:
        A dictionary where the keys are labels of track starts and the values
        are lists of labels belonging to the track.
    """
    parent_label, num_children = np.unique(tracks[:, 3], return_counts=True)
    num_children[parent_label == 0] = 0
    has_two_children = parent_label[num_children > 1]
    has_one_child = parent_label[num_children == 1]
    is_start = ~np.isin(tracks[:, 3], has_one_child)
    cluster = dict()
    for label in tracks[is_start, 0]:
        label_list = [label]
        current_label = label
        while True:
            if current_label in has_two_children:
                cluster[label] = label_list
                break
            elif current_label in has_one_child:
                idx = np.where(tracks[:, 3] == current_label)[0][0]
                current_label = tracks[idx, 0]
                label_list.append(current_label)
            else:
                cluster[label] = label_list
                break
    return cluster
