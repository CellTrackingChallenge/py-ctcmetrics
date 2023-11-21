import numpy as np
import tifffile as tiff
from sklearn.metrics import confusion_matrix
from scipy.sparse import lil_array


def match(ref_path, comp_path):
    """
    Matches the labels of the masks from the reference and computed result path.
    A label is matched if the intersection of a computed and reference mask is
    greater than 50% of the area of the reference mask.

    Args:
        ref_path: Path to the reference mask.
        comp_path: Path to the computed mask.

    Returns:
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
    intersection_over_ref[intersection_over_ref <= 0.5] = 0
    intersection_over_ref[intersection_over_ref > 0.5] = 1
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


def merge_tracks(
        comp_tracks,
        ref_tracks,
        labels_ref,
        labels_comp,
        mapped_ref,
        mapped_comp,
):
    """
    Merges tracklets that belong to the same real track. After this operation,
    two tracklets that belong to the same track but disappeared for some frames
    will have the same label. (Note that tracking file and masks are not
    consistent anymore!)

    Args:
        comp_tracks: The result tracks.
        ref_tracks: The ground truth tracks.
        labels_ref: The labels of the ground truth masks.
        labels_comp: The labels of the result masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The same as the input arguments but with merged tracks.
    """
    # Create an id mapping that is valid after merge
    ref_tracks = ref_tracks[np.argsort(ref_tracks[:, 1])]
    id_mapping_ref = {0: 0, -1: -1} | {x: x for x in ref_tracks[:, 0]}
    for i, _, _, p in ref_tracks:
        if p > 0:
            id_mapping_ref[i] = id_mapping_ref[p]
    comp_tracks = comp_tracks[np.argsort(comp_tracks[:, 1])]
    id_mapping_comp = {0: 0, -1: -1} | {x: x for x in comp_tracks[:, 0]}
    for i, _, _, p in comp_tracks:
        if p > 0:
            id_mapping_comp[i] = id_mapping_comp[p]
    # Remove unnecessary tracklets from track files
    ref_tracks[:, 0] = [id_mapping_ref[x] for x in ref_tracks[:, 0]]
    ref_tracks[:, 3] = [id_mapping_ref[x] for x in ref_tracks[:, 3]]
    uniques, counts = np.unique(ref_tracks[:, 0], return_counts=True)
    for i, c in zip(uniques, counts):
        if c == 1:
            continue
        inds = np.argwhere(ref_tracks[:, 0] == i)
        start, end = np.min(ref_tracks[inds, 1]), np.max(ref_tracks[inds, 2])
        ref_tracks = np.delete(ref_tracks, inds[1:], axis=0)
        ref_tracks[inds[0], 1] = start
        ref_tracks[inds[0], 2] = end
    comp_tracks[:, 0] = [id_mapping_comp[x] for x in comp_tracks[:, 0]]
    comp_tracks[:, 3] = [id_mapping_comp[x] for x in comp_tracks[:, 3]]
    uniques, counts = np.unique(comp_tracks[:, 0], return_counts=True)
    for i, c in zip(uniques, counts):
        if c == 1:
            continue
        inds = np.argwhere(comp_tracks[:, 0] == i)
        start, end = np.min(comp_tracks[inds, 1]), np.max(comp_tracks[inds, 2])
        comp_tracks = np.delete(comp_tracks, inds[1:], axis=0)
        comp_tracks[inds[0], 1] = start
        comp_tracks[inds[0], 2] = end
    # Rearrange track assignments
    labels_ref = [[id_mapping_ref[i] for i in l] for l in labels_ref]
    labels_comp = [[id_mapping_comp[i] for i in l] for l in labels_comp]
    mapped_ref = [[id_mapping_ref[i] for i in l] for l in mapped_ref]
    mapped_comp = [[id_mapping_comp[i] for i in l] for l in mapped_comp]

    return comp_tracks, ref_tracks, labels_ref, labels_comp, mapped_ref, \
        mapped_comp,


def count_acyclic_graph_correction_operations(
        gt_tracks,
        res_tracks,
        labels_gt,
        labels_res,
        mapped_gt,
        mapped_res,
):
    """
    Calculates the necessary operations to correct the result tracks to match
    the ground truth tracks. The operations are counted according to:
        Cell Tracking Accuracy Measurement Based on Comparison of Acyclic
        Oriented Graphs; Matula etal. 2015

    Args:
        comp_tracks: The result tracks.
        ref_tracks: The ground truth tracks.
        labels_ref: The labels of the ground truth masks.
        labels_comp: The labels of the result masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        NS, FN, FP, ED, EA, EC, num_vertices, num_edges
    """
    # Count vertices in the input data
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
        i_r = np.searchsorted(l_gt, m_gt, sorter=np.argsort(l_gt)) + ind_v_r
        i_c = np.searchsorted(l_res, m_res, sorter=np.argsort(l_res)) + ind_v_c
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
    comp, ref = det_test.nonzero()
    assert len(comp) == np.sum(V_tp_c)
    comp_to_ref, ref_to_comp = \
        np.zeros(num_V_C) * np.nan, np.zeros(num_V_R) * np.nan
    comp_to_ref[comp] = ref
    ref_to_comp[ref] = comp
    assert np.all(np.sort(comp) == comp)

    # Calculate relevant Vertex metrics
    TP = np.sum(V_tp_r)
    FN = np.sum(V_fn_r)
    FP = np.sum(V_fp_c)
    VS = np.sum(V_vs_c)
    NS = TP - (np.sum(V_tp_c) + VS)

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
        ind1 = np.argwhere(labels_gt[end1] == label1)[0] + cum_inds_R[end1]
        ind2 = np.argwhere(labels_gt[birth2] == label2)[0] + cum_inds_R[birth2]
        ind1, ind2 = int(ind1), int(ind2)
        t1, t2 = end1, birth2
        if parent2 in parents:
            semantic = 1
        elif parent2 in split_tracks:
            semantic = 1
        else:
            raise ValueError
        det_test1, det_test2 = int(V_tp_r[ind1]), int(V_tp_r[ind2])
        edges = np.asarray([
            ind1, label1, det_test1, t1,
            ind2, label2, det_test2, t2,
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
        ind1 = np.argwhere(labels_res[end1] == label1)[0] + cum_inds_C[end1]
        ind2 = np.argwhere(labels_res[birth2] == label2)[0] + cum_inds_C[birth2]
        ind1, ind2 = int(ind1), int(ind2)
        t1, t2 = end1, birth2
        if parent2 in parents:
            semantic = 1
        elif parent2 in split_tracks:
            semantic = 1
        else:
            raise ValueError
        det_test1, det_test2 = int(V_tp_c[ind1]), int(V_tp_c[ind2])
        edges = np.asarray([
            ind1, label1, det_test1, t1,
            ind2, label2, det_test2, t2,
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
        comp_to_ref[E_C_sub[:, 0]][:, None].astype(int),
        comp_to_ref[E_C_sub[:, 4]][:, None].astype(int)
    ], axis=1)
    assert not np.any(np.isnan(E_C_sub))

    # Map the edges to edges
    offset = 10 ** len(str(num_V_R))
    unique_edge_ids_R = E_R[:, 0] * offset + E_R[:, 4]
    unique_edge_ids_C = E_C_sub[:, 9] * offset + E_C_sub[:, 10]
    assert np.max(np.unique(unique_edge_ids_R, return_counts=True)[1]) == 1
    assert np.max(np.unique(unique_edge_ids_C, return_counts=True)[1]) == 1
    isin_R = np.isin(unique_edge_ids_C, unique_edge_ids_R)
    isin_C = np.isin(unique_edge_ids_R, unique_edge_ids_C)
    order_r = np.argsort(unique_edge_ids_R[isin_C])
    order_c = np.argsort(unique_edge_ids_C[isin_R])
    E_R_m = E_R[isin_C]
    E_C_sub_m = E_C_sub[isin_R]
    E_R_m = E_R_m[order_r]
    E_C_sub_m = E_C_sub_m[order_c]

    E_C_FP = E_C_sub[~isin_R]
    sem_different = E_C_sub_m[:, 8] != E_R_m[:, 8]
    E_C_CS = E_C_sub_m[sem_different]

    ED = len(E_C_FP)
    EA = np.sum(~isin_C)
    EC = len(E_C_CS)

    return NS, FN, FP, ED, EA, EC, num_V_R, len(E_R)


def assign_comp_to_ref(
        labels_ref,
        labels_comp,
        mapped_ref,
        mapped_comp
):
    """
    Assigns the computed labels to the reference labels.

    Args:
        labels_ref: The labels of the ground truth masks.
        labels_comp: The labels of the result masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        A dictionary with the reference labels as keys and the computed
        labels as values.
    """
    track_assignments = dict()
    for l_gt, l_res, m_gt, m_res in zip(
            labels_ref, labels_comp, mapped_ref, mapped_comp
    ):
        for i in l_gt:
            if i not in track_assignments:
                track_assignments[i] = list()
            if i in m_gt:
                match = m_res[int(np.argwhere(np.asarray(m_gt) == i)[0])]
                track_assignments[i].append(match)
            else:
                track_assignments[i].append(-1)
    track_assignments = {k: np.asarray(v) for k, v in track_assignments.items()}
    return track_assignments
