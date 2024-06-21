from test.utils import test_seq_res, test_seq_gt
from ctc_metrics import validate_sequence


def test_validate_sequence():
    res = validate_sequence(test_seq_res)
    assert bool(res["Valid"]) is True, f"{bool(res['Valid'])} != True"
