
def op_ctb(
        seg: float,
        tra: float,
):
    """
    Computes the OP_CTB metric. As described by
        [celltrackingchallenge](http://celltrackingchallenge.net/).

    Args:
        seg: The segmentation metric.
        tra: The tracking metric.

    Returns:
        The OP_CTB metric.
    """

    op = 0.5 * seg + 0.5 * tra
    return op
