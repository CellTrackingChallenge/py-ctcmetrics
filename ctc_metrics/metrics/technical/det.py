
def det(
    NS, FN, FP, num_vertices
):
    """
    Calculate Detection (DET) metric.

    According to
        Cell Tracking Accuracy Measurement Based on Comparison of Acyclic
        Oriented Graphs; Matula etal. 2015

    Args:
        NS: Split vertex operations.
        FN: Number of false negatives.
        FP: Number of false positives.
        num_vertices: Number of vertices in the graph.

    Returns:
        The Detection metric.

    """
    # Calculate AOGM_D
    w_ns = 5
    w_fn = 10
    w_fp = 1

    AOGM_D = w_ns * NS + w_fn * FN + w_fp * FP

    # Calculate AOGM_D0 (create graph from scratch)
    AOGM_D0 = w_fn * num_vertices  # All false negatives

    # Calculate DET
    DET = 1 - min(AOGM_D, AOGM_D0) / AOGM_D0

    return float(DET)

