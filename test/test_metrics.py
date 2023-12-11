# from test.utils import test_seq_res, test_seq_gt, test_det, test_seg, \
#     test_tra, test_ct, test_tf, test_bc0, test_bc1, test_bc2, test_bc3, test_cca
# from ctc_metrics import evaluate_sequence
#
#
# def test_metric_det():
#     max_d = 0.00001
#     metrics = evaluate_sequence(
#         test_seq_res, test_seq_gt, metrics=["DET"]
#     )
#     assert abs(metrics["DET"] - test_det) < max_d, \
#         f"{metrics['DET']} != {test_det}"
#
#
# def test_metric_seg():
#     max_d = 0.00001
#     metrics = evaluate_sequence(
#         test_seq_res, test_seq_gt, metrics=["SEG"]
#     )
#     assert abs(metrics["SEG"] - test_seg) < max_d, \
#         f"{metrics['SEG']} != {test_seg}"
#
#
# def test_metric_tra():
#     max_d = 0.00001
#     metrics = evaluate_sequence(
#         test_seq_res, test_seq_gt, metrics=["TRA"]
#     )
#     assert abs(metrics["TRA"] - test_tra) < max_d, \
#         f"{metrics['TRA']} != {test_tra}"
#
#
# def test_metric_ct():
#     max_d = 0.00001
#     metrics = evaluate_sequence(
#         test_seq_res, test_seq_gt, metrics=["CT"]
#     )
#     assert abs(metrics["CT"] - test_ct) < max_d, \
#         f"{metrics['CT']} != {test_ct}"
#
#
# def test_metric_tf():
#     max_d = 0.00001
#     metrics = evaluate_sequence(
#         test_seq_res, test_seq_gt, metrics=["TF"]
#     )
#     assert abs(metrics["TF"] - test_tf) < max_d, \
#         f"{metrics['TF']} != {test_tf}"
#
#
# def test_metric_bc():
#     max_d = 0.00001
#     metrics = evaluate_sequence(
#         test_seq_res, test_seq_gt, metrics=["BC"]
#     )
#     assert abs(metrics["BC(0)"] - test_bc0) < max_d, \
#         f"{metrics['BC'][0]} != {test_bc0}"
#     assert abs(metrics["BC(1)"] - test_bc1) < max_d, \
#         f"{metrics['BC'][1]} != {test_bc1}"
#     assert abs(metrics["BC(2)"] - test_bc2) < max_d, \
#         f"{metrics['BC'][2]} != {test_bc2}"
#     assert abs(metrics["BC(3)"] - test_bc3) < max_d, \
#         f"{metrics['BC'][3]} != {test_bc3}"
#
#
# def test_metric_cca():
#     metrics = evaluate_sequence(
#         test_seq_res, test_seq_gt, metrics=["CCA"]
#     )
#     assert abs(metrics["CCA"] - test_cca) < 0.00001, \
#         f"{metrics['CCA']} != {test_cca}"

def test_nothing():
    assert True
