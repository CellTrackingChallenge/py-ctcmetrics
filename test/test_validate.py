from ctc_metrics import  validate_sequence
from test.utils import test_seq_res, test_seq_gt


def test_validate_sequence():
    res = validate_sequence(test_seq_res, test_seq_gt)
    assert res["Valid"] == True, f"{res['Valid']} != True"
