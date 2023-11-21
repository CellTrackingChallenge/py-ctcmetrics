from ctc_metrics import evaluate_sequence, validate_sequence
from test.utils import test_root, test_seq_res, test_seq_gt, test_det, test_seg,\
    test_tra


def test_metrics():
    metrics = evaluate_sequence(test_seq_res, test_seq_gt)
    assert metrics["DET"] == test_det, f"{metrics['DET']} != {test_det}"
    assert metrics["SEG"] == test_seg, f"{metrics['SEG']} != {test_seg}"
    assert metrics["TRA"] == test_tra, f"{metrics['TRA']} != {test_tra}"
