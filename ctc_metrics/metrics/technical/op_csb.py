
def op_csb(
        seg: float,
        det: float,
):
    """
    Computes the OP_CSB metric. As described by
        [celltrackingchallenge](http://celltrackingchallenge.net/).

    Args:
        seg: The segmentation metric.
        det: The detection metric.

    Returns:
        The OP_CSB metric.
    """

    op = 0.5 * seg + 0.5 * det
    return op
