import numpy as np

from ctc_metrics.utils.representations import track_confusion_matrix


def mtml(
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list
):
    """
    Computes the mostly tracked and mostly lost metric. As described by
    [motchallenge](https://motchallenge.net/).

    Args:
        labels_comp: The labels of the computed masks.
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The mt and ml metrics.
    """
    # Gather association data
    track_intersection = track_confusion_matrix(
        labels_ref, labels_comp, mapped_ref, mapped_comp)
    # Calculate the metrics
    total_ref = np.sum(track_intersection[1:, :], axis=1)
    ratio = np.max(track_intersection[1:, :], axis=1) / np.maximum(total_ref, 1)
    valid_ref = total_ref > 0
    mt = np.sum(ratio[valid_ref] >= 0.8) / np.sum(valid_ref)
    ml = np.sum(ratio[valid_ref] < 0.2) / np.sum(valid_ref)

    res = {
        "MT": mt,
        "ML": ml,
    }
    return res
