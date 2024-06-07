import numpy as np

def faf(
        labels_comp: list,
        mapped_comp: list
):
    """
    Computes average number of false alarms per frame. As described by
    [motchallenge](https://motchallenge.net/).

    Args:
        labels_comp: The labels of the computed masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The FAF metric.
    """

    fp = 0
    frames = len(labels_comp)

    for comp, m_comp in zip(labels_comp, mapped_comp):
        uniques, counts = np.unique(m_comp, return_counts=True)
        uniques = uniques[counts == 1]
        fp += len(comp) - len(uniques)

    faf_score = fp / frames

    res = {
        "FAF": faf_score,
        "Frames": frames,
    }

    return res
