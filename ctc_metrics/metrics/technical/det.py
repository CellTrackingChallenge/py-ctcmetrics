import os
from os.path import join, split
import subprocess
import numpy as np
from scipy.sparse import coo_array, bsr_array, lil_array


def det_original(input_dir, num_digits):
    if os.name == 'nt':
        script_root = join(
            os.path.dirname(__file__), "third_party", "Win", "DETMeasure.exe"
        )
    elif os.name == 'posix':
        script_root = join(
            os.path.dirname(__file__), "third_party", "Linux", "DETMeasure")
    else:
        raise NotImplementedError

    if "01" in split(input_dir)[-1]:
        sequence = "01"
    elif "02" in split(input_dir)[-1]:
        sequence = "02"
    else:
        raise ValueError
    dataset_dir = join(*split(input_dir)[:-1])

    command = f"{script_root} " \
              f"{dataset_dir} " \
              f"{sequence} {num_digits}"
    try:
        out = subprocess.check_output(command, shell=True)
        det_measure = float(
            out.decode().replace("\n", "").replace("DET measure: ", ""))
    except Exception as e:
        print(
            "Error while processing DET measure with",
            command, e
        )
        det_measure = None

    return det_measure


def det(
    labels_gt,
    labels_res,
    mapped_gt,
    mapped_res,
):
    """
    According to
        Cell Tracking Accuracy Measurement Based on Comparison of Acyclic
        Oriented Graphs; Matula etal. 2015
    """
    # Create Vertex mapping
    num_V_R = np.sum([len(l) for l in labels_gt])
    num_V_C = np.sum([len(l) for l in labels_res])

    det_test = lil_array((num_V_C, num_V_R))
    ind_v_r = 0
    ind_v_c = 0
    for l_gt, l_res, m_gt, m_res in zip(
            labels_gt, labels_res, mapped_gt, mapped_res
    ):
        i_r = np.where(np.in1d(m_gt, l_gt))[0] + ind_v_r
        i_c = np.where(np.in1d(m_res, l_res))[0] + ind_v_c
        det_test[i_c, i_r] = 1
        ind_v_r += len(l_gt)
        ind_v_c += len(l_res)

    # Classify vertices to tp, fp, fn and vs
    assignments_r = np.sum(det_test, axis=0)
    assert np.max(assignments_r) <= 1
    V_tp_r = assignments_r == 1
    V_fn_r = ~V_tp_r

    assignments_c = np.sum(det_test, axis=1)
    V_tp_c = assignments_c == 1
    V_fp_c = assignments_c == 0

    # Calculate relevant metrics
    TP = np.sum(V_tp_r)
    FN = np.sum(V_fn_r)
    FP = np.sum(V_fp_c)
    NS = TP - np.sum(V_tp_c)

    # Calculate AOGM_D
    w_ns = 5
    w_fn = 10
    w_fp = 1
    w_ed = 0  # 1
    w_ea = 0  # 1.5
    w_ec = 0  # 1

    ED = EA = EC = 0

    AOGM_D = w_ns * NS + w_fn * FN + w_fp * FP + w_ed * ED + w_ea * EA + w_ec * EC

    # Calculate AOGM_D0 (create graph from scratch)
    AOGM_D0 = w_fn * num_V_R  # All false negatives

    # Calculate DET
    DET = 1 - min(AOGM_D, AOGM_D0) / AOGM_D0
    return DET, TP, FN, FP, NS

