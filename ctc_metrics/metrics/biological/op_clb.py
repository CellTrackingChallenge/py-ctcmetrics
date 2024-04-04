
def op_clb(
        lnk: float,
        bio: float,
):
    """
    Computes the OP_CLB metric. As described by
        [celltrackingchallenge](http://celltrackingchallenge.net/).

    Args:
        lnk: The linking metric.
        bio: The biological metric.

    Returns:
        The OP_CLB metric.
    """

    op = 0.5 * lnk + 0.5 * bio
    return op
