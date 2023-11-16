import os
from os.path import join, split
import subprocess
import numpy as np
from scipy.sparse import coo_array, bsr_array, lil_array


def tra_original(input_dir, num_digits):
    if os.name == 'nt':
        script_root = join(
            os.path.dirname(__file__), "third_party", "Win", "TRAMeasure.exe"
        )
    elif os.name == 'posix':
        script_root = join(
            os.path.dirname(__file__), "third_party", "Linux", "TRAMeasure")
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
        tra_measure = float(
            out.decode().replace("\n", "").replace("TRA measure: ", ""))
    except Exception as e:
        print(
            "Error while processing TRA measure with",
            command, e
        )
        tra_measure = None

    return tra_measure


def tra(
    gt_tracks,
    res_tracks,
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
    num_V_R = np.sum([len(l) for l in labels_gt])
    num_V_C = np.sum([len(l) for l in labels_res])
    cum_inds_R = np.cumsum([0] + [len(l) for l in labels_gt])
    cum_inds_C = np.cumsum([0] + [len(l) for l in labels_res])

    # Create Vertex mapping
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
    V_vs_c = assignments_c > 1

    # Mapping from reference to computed
    det_test[~V_tp_c, :] = 0
    det_test[:, ~V_tp_r] = 0
    comp, ref = det_test.nonzero()
    assert len(comp) == np.sum(V_tp_c)
    assert len(ref) == np.sum(V_tp_r)
    comp_to_ref, ref_to_comp = \
        np.zeros(num_V_C) * np.nan, np.zeros(num_V_R) * np.nan
    comp_to_ref[comp] = ref
    ref_to_comp[ref] = comp

    # Calculate relevant Vertex metrics
    TP = np.sum(V_tp_r)
    FN = np.sum(V_fn_r)
    FP = np.sum(V_fp_c)
    NS = TP - np.sum(V_tp_c)

    # Create edge mapping ...
    # ... for reference
    E_R = []  # ind1, id1, det_test1, t1, ind2, id2, det_test2, t2, semantic
    # Add track links
    ind_v_r = 0
    current_t = 0
    for l_gt1, l_gt2 in zip(labels_gt[:-1], labels_gt[1:]):
        l_gt1, l_gt2 = np.array(l_gt1), np.array(l_gt2)
        mapping = l_gt1[:, None] == l_gt2[None, :]
        ind1, ind2 = np.where(mapping)
        id1, id2 = l_gt1[ind1], l_gt2[ind2]
        t1, t2 = np.ones_like(id1) * current_t, np.ones_like(id1) * current_t + 1
        semantic = np.zeros_like(id1)
        ind1, ind2 = ind1 + ind_v_r, ind2 + ind_v_r + len(l_gt1)
        det_test1, det_test2 = V_tp_r[ind1], V_tp_r[ind2]
        edges = np.stack(
            [ind1, id1, det_test1, t1, ind2, id2, det_test2, t2, semantic],
            axis=1)
        E_R.append(edges)
        ind_v_r += len(l_gt1)
        current_t += 1
    # Add track and parent links from track file
    unique, counts = np.unique(gt_tracks[:, 3], return_counts=True)
    counts = counts[unique > 0]
    unique = unique[unique > 0]
    parents = unique[counts > 1]
    split_tracks = unique[counts == 1]
    for track in gt_tracks:
        label2, birth2, end2, parent2 = track
        if parent2 == 0:
            continue
        label1, birth1, end1, parent1 = gt_tracks[gt_tracks[:, 0] == parent2][0]
        ind1 = np.argwhere(labels_gt[birth1] == label1)[0] + cum_inds_R[birth1]
        ind2 = np.argwhere(labels_gt[birth2] == label2)[0] + cum_inds_R[birth2]
        t1, t2 = birth1, birth2
        if parent2 in parents:
            semantic = 1
        elif parent2 in split_tracks:
            semantic = 0
        else:
            raise ValueError
        det_test1, det_test2 = V_tp_r[ind1], V_tp_r[ind2]
        edges = np.asarray([
            ind1[0], label1, int(det_test1[0]), t1,
            ind2[0], label2, int(det_test2[0]), t2,
            semantic],
        )[None, :]
        E_R.append(edges)
    E_R = np.concatenate(E_R, axis=0)

    # ... for computed
    E_C = []  # ind1, id1, det_test1, t1, ind2, id2, det_test2, t2, semantic
    ind_v_c = 0
    current_t = 0
    for l_res1, l_res2 in zip(labels_res[:-1], labels_res[1:]):
        l_res1, l_res2 = np.array(l_res1), np.array(l_res2)
        mapping = l_res1[:, None] == l_res2[None, :]
        ind1, ind2 = np.where(mapping)
        id1, id2 = l_res1[ind1], l_res2[ind2]
        t1, t2 = np.ones_like(id1) * current_t, np.ones_like(id1) * current_t + 1
        semantic = np.zeros_like(id1)
        ind1, ind2 = ind1 + ind_v_c, ind2 + ind_v_c + len(l_res1)
        det_test1, det_test2 = V_tp_c[ind1], V_tp_c[ind2]
        edges = np.stack(
            [ind1, id1, det_test1, t1, ind2, id2, det_test2, t2, semantic],
            axis=1)
        E_C.append(edges)
        ind_v_c += len(l_res1)
        current_t += 1
    # Add track and parent links from track file
    unique, counts = np.unique(res_tracks[:, 3], return_counts=True)
    counts = counts[unique > 0]
    unique = unique[unique > 0]
    parents = unique[counts > 1]
    split_tracks = unique[counts == 1]
    for track in res_tracks:
        label2, birth2, end2, parent2 = track
        if parent2 == 0:
            continue
        label1, birth1, end1, parent1 = \
            res_tracks[res_tracks[:, 0] == parent2][0]
        ind1 = np.argwhere(labels_res[birth1] == label1)[0] + cum_inds_C[birth1]
        ind2 = np.argwhere(labels_res[birth2] == label2)[0] + cum_inds_C[birth2]
        t1, t2 = birth1, birth2
        if parent2 in parents:
            semantic = 1
        elif parent2 in split_tracks:
            semantic = 0
        else:
            raise ValueError
        det_test1, det_test2 = V_tp_c[ind1], V_tp_c[ind2]
        edges = np.asarray([
            ind1[0], label1, int(det_test1[0]), t1,
            ind2[0], label2, int(det_test2[0]), t2,
            semantic],
        )[None, :]
        E_C.append(edges)
    E_C = np.concatenate(E_C, axis=0)

    # Create the induced subgraph with only uniquely matched vertices
    E_C_sub = E_C[(E_C[:, 2] * E_C[:, 6]) == 1]
    # Add mapping to Reference graph such that E_C_sub is:
    #    ind1,id1,det_tst1,t1,ind2,id2,det_test2,t2,sem,ind1_R,ind2_R
    E_C_sub = np.concatenate([
        E_C_sub,
        comp_to_ref[E_C_sub[:, 0]][:,None],
        comp_to_ref[E_C_sub[:, 4]][:,None]
    ], axis=1)
    assert not np.any(np.isnan(E_C_sub))

    # Map the edges to edges
    offset = 10 ** len(str(num_V_R))
    unique_edge_ids_R = E_R[:, 0] * offset + E_R[:, 4]
    unique_edge_ids_C = E_C_sub[:, 9] * offset + E_C_sub[:, 10]
    assert np.max(np.unique(unique_edge_ids_R,return_counts=True)[1]) == 1
    assert np.max(np.unique(unique_edge_ids_C,return_counts=True)[1]) == 1
    isin_R = np.isin(unique_edge_ids_C, unique_edge_ids_R)
    mapping = np.searchsorted(
        unique_edge_ids_R, unique_edge_ids_C[isin_R],
        sorter=np.argsort(unique_edge_ids_R))
    assert np.max(np.unique(mapping,return_counts=True)[1]) == 1
    E_C_FP = E_C_sub[~isin_R]
    sem_different = E_C_sub[isin_R, 8] != E_R[mapping.astype(int), 8]
    E_C_CS = E_C_sub[isin_R][sem_different]

    # Calculate relevant edge metrics
    ED = len(E_C_FP)
    EA = len(E_R) - np.sum(isin_R)
    EC = len(E_C_CS)

    # Calculate AOGM
    w_ns = 5
    w_fn = 10
    w_fp = 1
    w_ed = 1
    w_ea = 1.5
    w_ec = 1

    AOGM = w_ns * NS + w_fn * FN + w_fp * FP + w_ed * ED + w_ea * EA + w_ec * EC

    # Calculate AOGM_0 (create graph from scratch)
    #   i.e, all vertices and edges are false negatives
    AOGM_0 = w_fn * num_V_R + w_ea * len(E_R)

    # Calculate DET
    TRA = 1 - min(AOGM, AOGM_0) / AOGM_0
    return TRA, NS, FN, FP, ED, EA, EC


if __name__ == "__main__":
    a = [1, 2, 3]
    b = [-2, -1, 0, 1, 2, 3, 4]
    print(np.searchsorted(b, a))
