
def tra(
    NS, FN, FP, ED, EA, EC, num_vertices, num_edges
):
    """
    Calculate Tracking (TRA) metric.

    According to
        Cell Tracking Accuracy Measurement Based on Comparison of Acyclic
        Oriented Graphs; Matula etal. 2015

    Args:
        NS: Split vertex operations.
        FN: Number of false negatives.
        FP: Number of false positives.
        ED: Number of redundant edge.
        EA: Number of missing edges.
        EC: Number of wrong edge semantics.
        num_vertices: Number of vertices in the graph.
        num_edges: Number of edges in the graph.

    Returns:
        The Tracking metric.
    """
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
    AOGM_0 = w_fn * num_vertices + w_ea * num_edges

    # Calculate DET
    TRA = 1 - min(AOGM, AOGM_0) / AOGM_0
    return float(TRA)

