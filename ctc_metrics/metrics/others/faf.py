import numpy as np

from ctc_metrics.utils.representations import track_confusion_matrix


def faf(
        labels_comp: list,
        mapped_comp: list
):
    """
    Computes average number of false alarms per frame. As described by
    [motchallenge](https://motchallenge.net/).

    Args:
        labels_comp: The labels of the computed masks.
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The faf metric.
    """

    fp = 0
    frames = len(labels_comp)

    for comp, m_comp in zip(labels_comp, mapped_comp):
        uniques, counts = np.unique(m_comp, return_counts=True)
        uniques = uniques[counts == 1]
        fp += len(comp) - len(uniques)

    faf = fp / frames

    res = {
        "FAF": faf,
        "Frames": frames,
    }

    return res
