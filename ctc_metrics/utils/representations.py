import numpy as np
import tifffile as tiff
from sklearn.metrics import confusion_matrix


def match(ref_path, comp_path):
    """
    Matches the labels of two masks.

    args:
        ref_path: Path to the reference mask.
        comp_path: Path to the computed mask.

    returns:
        A tuple of five numpy arrays. The first array contains the existing
        labels in the reference mask. The second array contains the
        existing labels in the computed mask. The third array contains the
        matched labels in the referenced mask. The fourth array contains
        the corresponding matched labels in the computed mask. The fifth
        array contains the intersection over union (IoU) for each matched
        label pair.
    """
    # Read the input data
    if ref_path is None:
        # For trivial cases where only one mask should be analysed
        ref_path = comp_path
    map_ref = tiff.imread(ref_path)
    map_com = tiff.imread(comp_path)
    # Get the labels of the two masks (including background label 0)
    labels_ref, labels_comp = np.unique(map_ref), np.unique(map_com)
    if ref_path == comp_path:
        # For trivial cases where only one mask should be analysed
        iou = np.ones(len(labels_ref))
        return labels_ref.tolist(), labels_comp.tolist(), labels_ref.tolist(),\
            labels_comp.tolist(), iou.tolist()
    # Add offset to separate the labels of the two masks
    offset = int(np.max(labels_ref) + 1)
    map_com += offset
    # Compute the confusion matrix
    cm = confusion_matrix(map_ref.flatten(), map_com.flatten())
    sum_ref = np.sum(cm, axis=1, keepdims=True)
    sum_comp = np.sum(cm, axis=0, keepdims=True)
    # Compute the intersection over reference
    intersection_over_ref = cm / np.maximum(sum_ref, 1)
    # Compute the intersection over union (relevant to calculate SEG)
    intersection_over_union = cm / np.maximum(sum_ref + sum_comp - cm, 1)
    # Remove the background label and redundant parts of the matrix
    intersection_over_ref = \
        intersection_over_ref[1:len(labels_ref), 1 + len(labels_ref):]
    intersection_over_union = \
        intersection_over_union[1:len(labels_ref), 1 + len(labels_ref):]
    # Find matches according to AOGM (min 50% of ref needs to be covered)
    intersection_over_ref[intersection_over_ref < 0.5] = 0
    intersection_over_ref[intersection_over_ref >= 0.5] = 1
    # Create mapping between reference and computed labels
    rows, cols = np.nonzero(intersection_over_ref)
    labels_ref = labels_ref[1:]
    labels_comp = labels_comp[1:]
    mapped_ref = labels_ref[rows].tolist()
    mapped_comp = labels_comp[cols].tolist()
    iou = intersection_over_union[rows, cols].tolist()
    assert np.unique(mapped_ref).size == len(mapped_ref), \
        f"Reference node assigned to multiple computed nodes! " \
        f"{ref_path} {labels_ref, labels_comp, mapped_ref, mapped_comp}"
    labels_ref = labels_ref.tolist()
    labels_comp = labels_comp.tolist()
    return labels_ref, labels_comp, mapped_ref, mapped_comp, iou

# def match(gt_mask_path, res_mask_path):
#     """
#     Matches the labels of two masks.
#
#     # Checked against test datasets -> OK
#
#     args:
#         gt_mask_path: Path to the ground truth mask.
#         res_mask_path: Path to the result mask.
#
#     returns:
#         A tuple of four numpy arrays. The first array contains the existing
#         labels in the ground truth mask. The second array contains the
#         existing labels in the result mask. The third array contains the
#         matched labels in the ground truth mask. The fourth array contains
#         the corresponding matched labels in the result mask.
#     """
#     if gt_mask_path is None:
#         gt_mask_path = res_mask_path
#     map1 = tiff.imread(gt_mask_path)
#     map2 = tiff.imread(res_mask_path)
#     # Calculate the IoU matrix of the two masks
#     labels1, labels2 = np.unique(map1), np.unique(map2)
#     if gt_mask_path == res_mask_path:
#         return labels1, labels2, labels1, labels2
#     offset = int(np.max(labels1) + 1)
#     map2 += offset
#     cm = confusion_matrix(map1.flatten(), map2.flatten())
#     cm[0, :] = 0
#     norm1 = cm / (np.sum(cm, axis=1, keepdims=True) + 0.0000001)
#     norm2 = cm / (np.sum(cm, axis=0, keepdims=True) + 0.0000001)
#     iogt = norm1 * norm2
#     iogt[len(labels1):] = 0
#     iogt[:, :len(labels1)+1] = 0
#     iogt = iogt[1:len(labels1), 1 + len(labels1):]
#     # Suppress matches all non-maxima with respect to the ground truth
#     iogt[iogt < 0.5] = 0
#     iogt[iogt >= 0.5] = 1
#     # Remove double assigned detections
#     col_sum = np.sum(iogt, axis=0, keepdims=False)
#     iogt[:, col_sum > 1] = 0
#     # Create Mapping
#     rows, cols = np.nonzero(iogt)
#     labels1 = labels1[1:]
#     labels2 = labels2[1:]
#     mapped1 = labels1[rows].tolist()
#     mapped2 = labels2[cols].tolist()
#     assert np.unique(mapped1).size == len(mapped1), f"a {gt_mask_path} {len(labels1), len(mapped1), iogt.shape, cm.shape, iogt}"
#     assert np.unique(mapped2).size == len(mapped2), f"b {gt_mask_path} {len(labels1), len(mapped1), iogt.shape, cm.shape, labels1,labels2, mapped1, mapped2, rows, cols}"
#     labels1 = labels1.tolist()
#     labels2 = labels2.tolist()
#     return labels1, labels2, mapped1, mapped2


def cluster_full_tracks(tracks):
    """ Clusters tracks list into clusters that belong to the same real track.

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
