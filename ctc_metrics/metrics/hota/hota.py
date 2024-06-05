import numpy as np

from ctc_metrics.utils.representations import track_confusion_matrix


def hota(
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list,
):
    """
    Computes the HOTA metric. As described in the paper,
         "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking"
           - Luiten et al., {International Journal of Computer Vision 2020

    Args:
        labels_comp: The labels of the computed masks.
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The hota tracks metric.
    """

    max_label_ref = int(np.max(np.concatenate(labels_ref)))
    max_label_comp = int(np.max(np.concatenate(labels_comp)))

    # Gather association data
    track_intersection = track_confusion_matrix(
        labels_ref, labels_comp, mapped_ref, mapped_comp)

    # Calculate Association scores
    hota_score = 0
    for i in range(1, max_label_ref + 1):
        for j in range(1, max_label_comp + 1):
            if track_intersection[i, j] > 0:
                # Calculate the HOTA score
                tpa = track_intersection[i, j]
                fna = np.sum(track_intersection[i, :]) - tpa
                fpa = np.sum(track_intersection[:, j]) - tpa
                a_corr = tpa / (tpa + fna + fpa)
                hota_score += tpa * a_corr

    tp = track_intersection[1:, 1:].sum()
    fp = track_intersection[0, 1:].sum()
    fn = track_intersection[1:, 0].sum()
    hota_score = np.sqrt(hota_score / (tp + fp + fn))

    res = {
        "HOTA": hota_score,
    }
    return res
