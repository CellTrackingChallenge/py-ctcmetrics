import numpy as np

from ctc_metrics.metrics import raw_division_metrics
from ctc_metrics.utils.representations import assign_comp_to_ref


def test_assign_comp_to_ref():
    # Frame 0: reference 1 -> 5, 2 -> 6, reference 3 is not matched.
    # Frame 1: computed 5 is matched to references 1 and 2 (not unique), 3 -> 7.
    labels_ref = [[1, 2, 3], [1, 2, 3]]
    mapped_ref = [[1, 2], [1, 2, 3]]
    mapped_comp = [[5, 6], [5, 5, 7]]
    assignments = assign_comp_to_ref(labels_ref, mapped_ref, mapped_comp)
    assert sorted(assignments) == [1, 2, 3]
    np.testing.assert_array_equal(assignments[1], [5, 0])
    np.testing.assert_array_equal(assignments[2], [6, 0])
    np.testing.assert_array_equal(assignments[3], [0, 7])


def test_raw_division_metrics():
    # Tracks: label, birth frame, end frame, parent. Two divisions in each.
    ref_tracks = np.array([
        [1, 0, 1, 0], [2, 2, 3, 1], [3, 2, 3, 1],
        [4, 0, 1, 0], [5, 2, 3, 4], [6, 2, 3, 4],
    ])
    comp_tracks = np.array([
        [10, 0, 1, 0], [11, 2, 3, 10], [12, 2, 3, 10],
        [20, 0, 1, 0], [21, 2, 3, 20], [22, 2, 3, 20],
    ])
    # Division 1 -> 2, 3 is found as 10 -> 11, 12. Division 4 -> 5, 6 is not:
    # its computed counterpart 20 has the daughter 21 (= 5), but 22 matches
    # no reference daughter and 6 is not matched.
    mapped_ref = [[1, 4], [1, 4], [2, 3, 5], [2, 3, 5]]
    mapped_comp = [[10, 20], [10, 20], [11, 12, 21], [11, 12, 21]]
    tp, fp, fn = raw_division_metrics(comp_tracks, ref_tracks, mapped_ref, mapped_comp, 0)
    assert (tp, fp, fn) == (1, 1, 1)
