
def lnk(
        ED: int,
        EA: int,
        EC: int,
        num_edges: int,
        **_ignored
):
    """
    Calculate Linking (LNK) metric.

    According to
        Cell Tracking Accuracy Measurement Based on Comparison of Acyclic
        Oriented Graphs; Matula etal. 2015

    Args:
        ED: Number of redundant edge.
        EA: Number of missing edges.
        EC: Number of wrong edge semantics.
        num_edges: Number of edges in the graph.
        _ignored: Ignored arguments.

    Returns:
        The Linking metric.
    """
    # Calculate AOGM_A
    w_ed = 1
    w_ea = 1.5
    w_ec = 1
    AOGM_A = w_ed * ED + w_ea * EA + w_ec * EC
    # Calculate AOGM_0 (create graph from scratch)
    #   i.e, all vertices and edges are false negatives
    AOGM_0 = w_ea * num_edges
    # Calculate DET
    LNK = 1 - min(AOGM_A, AOGM_0) / AOGM_0
    return float(LNK)

