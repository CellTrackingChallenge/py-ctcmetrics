# pylint: disable=no-name-in-module

import os.path

import numpy as np
import tifffile as tiff
from sklearn.metrics import confusion_matrix
from scipy.sparse import lil_array  # noqa


def track_confusion_matrix(
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list
):
    """
    Computes the confusion matrix for the input data.

    Args:
        labels_ref: The labels of the ground truth masks.
        labels_comp: The labels of the result masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The confusion matrix. The size if the confusion matrix is
            max_label_ref + 1 x max_label_comp + 1
        where [:, 0] contains the false negatives and [0, :] contains the
        false positives. The rest of the matrix contains the true positives.

    """
    max_label_ref = int(np.max(np.concatenate(labels_ref)))
    max_label_comp = int(np.max(np.concatenate(labels_comp)))

    # Gather association data
    track_intersection = np.zeros((max_label_ref + 1, max_label_comp + 1))

    for ref, comp, m_ref, m_comp in zip(
            labels_ref, labels_comp, mapped_ref, mapped_comp):
        # Fill track intersection matrix
        ref = np.asarray(ref)
        comp = np.asarray(comp).astype(int)
        m_ref = np.asarray(m_ref).astype(int)
        m_comp = np.asarray(m_comp).astype(int)
        if len(m_ref) > 0:
            track_intersection[m_ref, m_comp] += 1
        fna = ref[np.isin(ref, m_ref, invert=True)]
        track_intersection[fna, 0] += 1
        fpa = comp[np.isin(comp, m_comp, invert=True)].astype(int)
        track_intersection[0, fpa] += 1

    return track_intersection


def match(
        ref_path: str,
        comp_path: str
):
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
    if comp_path is None:
        map_ref = tiff.imread(ref_path)
        labels_ref = np.unique(map_ref)
        labels_ref = labels_ref[labels_ref != 0]
        return labels_ref.tolist(), [], [], [], []
    if ref_path is None:
        # For trivial cases where only one mask should be analysed
        ref_path = comp_path
    if os.path.exists(ref_path):
        # No slices
        map_ref = tiff.imread(ref_path)
        map_com = tiff.imread(comp_path)
    else:
        # Slices in 3D dataset segmentation data
        dirname, base = os.path.dirname(ref_path), os.path.basename(ref_path)
        base = base.replace(".tif", "")
        slice_files = [x for x in os.listdir(dirname) if
                 x.endswith(".tif") and x.startswith(base)]
        slice_files = sorted(slice_files)
        map_ref = [tiff.imread(os.path.join(dirname, x)) for x in slice_files]
        map_ref = np.stack(map_ref, axis=0)
        slices = [x.replace(base+"_", "").replace(".tif", "")
                  for x in slice_files]
        slices = [int(x) for x in slices]
        map_com = tiff.imread(comp_path)
        _map_com = [map_com[x] for x in slices]
        map_com = np.stack(_map_com, axis=0)
    # Get the labels of the two masks (including background label 0)
    labels_ref, labels_comp = np.unique(map_ref), np.unique(map_com)
    if ref_path == comp_path:
        # For trivial cases where only one mask should be analysed
        iou = np.ones(len(labels_ref))
        labels_ref = labels_ref[labels_ref != 0]
        labels_comp = labels_comp[labels_comp != 0]
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


def create_edge_mapping(
        tracks: np.ndarray,
        labels: list,
        V_tp: np.ndarray,
        cum_inds: np.ndarray,
):
    """
    Creates the edge mapping for the input tracks. The edge mapping is a
    nd.array with the following style:
        [[ind1, id1, det_test1, t1, ind2, id2, det_test2, t2, semantic]]
    where ind1 and ind2 are the indices of the nodes in V_tp, id1 and
    id2 are the labels of the vertices, det_test1 and det_test2 are the
    detection test of the vertices, t1 and t2 are the time steps of the
    vertices and semantic is the semantic label of the edge (0 for track link,
    1 for parent link).

    Args:
        tracks: The tracks. If None, no edges will be created
        labels: The labels of the ground truth masks.
        V_tp: The detection test matrix.
        cum_inds: The cumulative indices of the vertices per frame.

    Returns:
        The edge mapping.
    """
    if tracks is None:
        return np.zeros((0, 9))
    all_edges = []
    # Add track links
    ind_v = 0
    current_t = 0
    for l_gt1, l_gt2 in zip(labels[:-1], labels[1:]):
        l_gt1, l_gt2 = np.array(l_gt1), np.array(l_gt2)
        mapping = l_gt1[:, None] == l_gt2[None, :]
        ind1, ind2 = np.where(mapping)
        id1, id2 = l_gt1[ind1], l_gt2[ind2]
        t1, t2 = np.ones_like(id1) * current_t, np.ones_like(id1) * current_t + 1
        ind1, ind2 = ind1 + ind_v, ind2 + ind_v + len(l_gt1)
        edges = np.stack([
            ind1, id1, V_tp[ind1], t1,
            ind2, id2,  V_tp[ind2], t2,
            np.zeros_like(id1)], axis=1)
        all_edges.append(edges)
        ind_v += len(l_gt1)
        current_t += 1
    # Add track and parent links from track file
    for track in tracks:
        label2, birth2, _, parent2 = track
        if parent2 == 0:
            continue
        label1, _, end1, _ = tracks[tracks[:, 0] == parent2][0]
        ind1 = np.argwhere(labels[end1] == label1)[0] + cum_inds[end1]
        ind2 = np.argwhere(labels[birth2] == label2)[0] + cum_inds[birth2]
        ind1, ind2 = int(ind1), int(ind2)
        edges = np.asarray(
            [ind1, label1, int(V_tp[ind1]), end1, ind2, label2, int(V_tp[ind2]), birth2, 1]
        )[None, :]
        all_edges.append(edges)
    return np.concatenate(all_edges, axis=0).astype(int)


def create_detection_test_matrix(
        num_V_C: int,
        num_V_R: int,
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list,
):
    """
    Creates the detection test matrix for the input tracks. The detection test
    is stored as a sparse matrix with num_V_C rows and num_V_R columns. The
    matrix is filled with 1 if the vertex is a match and 0 otherwise.

    Args:
        num_V_C: The number of vertices in the computed graph.
        num_V_R: The number of vertices in the reference graph.
        labels_ref: The labels of the ground truth masks.
        labels_comp: The labels of the result masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        The detection test matrix.
    """
    det_test = lil_array((num_V_C, num_V_R))
    ind_v_r = 0
    ind_v_c = 0
    for l_gt, l_res, m_gt, m_res in zip(
            labels_ref, labels_comp, mapped_ref, mapped_comp
    ):
        i_r = np.searchsorted(l_gt, m_gt, sorter=np.argsort(l_gt)) + ind_v_r
        i_c = np.searchsorted(l_res, m_res, sorter=np.argsort(l_res)) + ind_v_c
        det_test[i_c, i_r] = 1
        ind_v_r += len(l_gt)
        ind_v_c += len(l_res)
    return det_test


def count_acyclic_graph_correction_operations(
        ref_tracks: np.ndarray,
        comp_tracks: np.ndarray,
        labels_ref: list,
        labels_comp: list,
        mapped_ref: list,
        mapped_comp: list,
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
    stats = {}
    stats["num_vertices_R"] = np.sum([len(l) for l in labels_ref])
    stats["num_vertices_C"] = np.sum([len(l) for l in labels_comp])
    stats["num_vertices"] = stats["num_vertices_R"]
    # Cumulate the number of vertices per frame
    cum_inds_R = np.cumsum([0] + [len(l) for l in labels_ref])
    cum_inds_C = np.cumsum([0] + [len(l) for l in labels_comp])
    # Perform "detection test"
    det_test = create_detection_test_matrix(
        stats["num_vertices_C"], stats["num_vertices_R"],
        labels_ref, labels_comp, mapped_ref, mapped_comp
    )
    # Classify vertices to tp, fp, fn and vs
    assignments_r = np.sum(det_test, axis=0)
    assignments_c = np.sum(det_test, axis=1)
    assert np.max(assignments_r) <= 1
    V_tp_r = assignments_r == 1
    V_tp_c = assignments_c == 1
    stats["TP"] = np.sum(V_tp_r)
    stats["FN"] = np.sum(~V_tp_r)
    stats["FP"] = np.sum(assignments_c == 0)
    stats["VS"] = np.sum(assignments_c > 1)
    stats["NS"] = stats["TP"] - (np.sum(V_tp_c) + stats["VS"])
    # Mapping from reference to computed
    det_test[~V_tp_c, :] = 0
    comp, ref = det_test.nonzero()
    assert len(comp) == np.sum(V_tp_c)
    comp_to_ref = np.zeros(stats["num_vertices_C"]) * np.nan
    comp_to_ref[comp] = ref
    assert np.all(np.sort(comp) == comp)
    # Create edge mapping ...
    # ... for reference
    E_R = create_edge_mapping(ref_tracks, labels_ref, V_tp_r, cum_inds_R)
    # ... for computed
    E_C = create_edge_mapping(comp_tracks, labels_comp, V_tp_c, cum_inds_C)
    # Stop calculation if no edges are present in the computed graph (e.g. if
    #   only segmentation is present)
    if E_C.size == 0:
        stats["ED"] = 0
        stats["EA"] = len(E_R)
        stats["EC"] = 0
        stats["num_edges"] = len(E_R)
        return stats
    # Reduce the computed graph to an induced subgraph with only uniquely
    #   matched vertices
    E_C = E_C[(E_C[:, 2] * E_C[:, 6]) == 1]
    # Add mapping to Reference graph such that E_C is:
    #    ind1,id1,det_tst1,t1,ind2,id2,det_test2,t2,sem,ind1_R,ind2_R
    E_C = np.concatenate([
        E_C,
        comp_to_ref[E_C[:, 0]][:, None].astype(int),
        comp_to_ref[E_C[:, 4]][:, None].astype(int)
    ], axis=1)
    assert not np.any(np.isnan(E_C))
    # Map the edges to edges
    unique_edge_ids_R = (E_R[:, 0] * 10 ** len(str(stats["num_vertices_R"]))
                         + E_R[:, 4])
    unique_edge_ids_C = (E_C[:, 9] * 10 ** len(str(stats["num_vertices_R"]))
                         + E_C[:, 10])
    if unique_edge_ids_R.size > 0:
        assert np.max(np.unique(unique_edge_ids_R, return_counts=True)[1]) == 1
    if unique_edge_ids_C.size > 0:
        assert np.max(np.unique(unique_edge_ids_C, return_counts=True)[1]) == 1
    isin_R = np.isin(unique_edge_ids_C, unique_edge_ids_R)
    isin_C = np.isin(unique_edge_ids_R, unique_edge_ids_C)
    E_R_mapped = E_R[isin_C]
    E_C_mapped = E_C[isin_R]
    E_R_mapped = E_R_mapped[np.argsort(unique_edge_ids_R[isin_C])]
    E_C_mapped = E_C_mapped[np.argsort(unique_edge_ids_C[isin_R])]
    # Calculate relevant edge statistics
    stats["ED"] = len(E_C[~isin_R])
    stats["EA"] = np.sum(~isin_C)
    stats["EC"] = len(E_C_mapped[E_C_mapped[:, 8] != E_R_mapped[:, 8]])
    stats["num_edges"] = len(E_R)
    return stats


def assign_comp_to_ref(
        labels_ref: list,
        mapped_ref: list,
        mapped_comp: list,
):
    """
    Assigns the computed labels to the reference labels.

    Args:
        labels_ref: The labels of the ground truth masks.
        mapped_ref: The matched labels of the ground truth masks.
        mapped_comp: The matched labels of the result masks.

    Returns:
        A dictionary with the reference labels as keys and the computed
        labels as values.
    """
    all_labels = np.unique(np.concatenate(labels_ref))
    max_frame = len(labels_ref)
    track_assignments = {
        k: np.zeros(max_frame) * np.nan for k in all_labels
    }
    frame = 0
    for l_gt, m_gt, m_res in zip(
            labels_ref, mapped_ref, mapped_comp
    ):
        for i in l_gt:
            if i in m_gt:
                m = m_res[int(np.argwhere(np.asarray(m_gt) == i)[0])]
                track_assignments[i][frame] = m
                counts = np.sum(np.asarray(m_res) == m)
                if counts > 1:
                    track_assignments[i][frame] = 0
            else:
                track_assignments[i][frame] = 0
        frame += 1
    return track_assignments


def merge_tracks(
        tracks: np.ndarray,
        labels: list,
        mapped: list
):
    """
    Merges tracks that belong to the same cell trajectory. This can happen, if
    a cell is invisible for a few frames and the tracking algorithm splits the
    track into two or more tracks. The resulting data is a relabelled version
    of the input data.

    Args:
        tracks: The tracks.
        labels: The labels of the masks.
        mapped: The matched labels of the masks.

    Returns:
        As input data, but relabelled.
    """
    tracks = np.copy(tracks)
    labels = [np.copy(x) for x in labels]
    mapped = [np.copy(x) for x in mapped]

    # Find tracks that belong together
    parents, cnts = np.unique(tracks[tracks[:, 3] > 0, 3], return_counts=True)
    parents = parents[cnts == 1].tolist()
    children = [tracks[tracks[:, 3] == p][0, 0] for p in parents]

    # Merge tracks
    mapping = {x: x for x in np.unique(tracks[:, 0])}
    while len(children) > 0:
        for parent, child in zip(parents, children):
            if parent not in children:
                break
            mapping[child] = mapping[parent]
            children.remove(child)
            parents.remove(parent)

    # Relabel such that the labels are continuous
    remaining_labels = sorted(np.unique(list(mapping.values())))
    new_labels = list(range(1, len(remaining_labels) + 1))
    for k, v in mapping.items():
        new_v = new_labels[remaining_labels.index(v)]
        mapping[k] = new_v

    lut = np.zeros(np.max(list(mapping.keys())) + 1, dtype=int)
    for k, v in mapping.items():
        lut[k] = v

    # Relabel tracks
    new_tracks = np.copy(tracks)
    for k, v in mapping.items():
        new_tracks[tracks[:, 0] == k, 0] = v
        new_tracks[tracks[:, 3] == k, 3] = v
    ids, counts = np.unique(new_tracks[:, 0], return_counts=True)
    ids = ids[counts > 1]
    for i in ids:
        inds = np.argwhere(new_tracks[:, 0] == i).flatten()
        start = np.min(new_tracks[inds, 1])
        end = np.max(new_tracks[inds, 2])
        new_tracks[inds, 1] = start
        new_tracks[inds, 2] = end
        new_tracks = np.delete(new_tracks, inds[1:], axis=0)

    # Relabel labels and mapped
    new_labels = [[lut[x] for x in i] for i in labels]
    new_mapped = [[lut[x] for x in i] for i in mapped]

    return new_tracks, new_labels, new_mapped




