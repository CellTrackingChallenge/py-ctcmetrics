import numpy as np


def mota(
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list
):
    """
    Computes the MOTA metric. As described in the paper,
         "Evaluating Multiple Object Tracking Performance:
          The CLEAR MOT Metrics."
           - Keni Bernardin and Rainer Stiefelhagen, EURASIP 2008

    Args:
        labels_comp: The labels of the computed masks.
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The mota tracks metric.
    """
    tp, fp, fn, idsw, multi_assignments = 0, 0, 0, 0, 0

    max_label_gt = int(np.max(np.concatenate(labels_ref)))
    matches = np.zeros(max_label_gt + 1)

    for ref, comp, m_ref, m_comp in zip(
            labels_ref, labels_comp, mapped_ref, mapped_comp):
        # Calculate metrics
        labels, counts = np.unique(m_comp, return_counts=True)
        tp += len(m_ref)
        fn += len(ref) - len(m_ref)
        fp += len(comp) - len(m_comp) + np.sum(counts[counts > 1] - 1)
        multi_assignments += np.sum(counts[counts > 1] - 1)
        idsw += np.sum(matches[m_ref] != m_comp)
        # Update the match cache
        matches[m_ref] = m_comp

    mota = 1 - (fn + fp + idsw + multi_assignments) / (tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    res = {
        "MOTA": mota,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "IDSW": idsw,
        "MULTI-ASSIGNMENTS": multi_assignments,
        "Precision": precision,
        "Recall": recall
    }
    return res
