import argparse
from os.path import join, basename
from multiprocessing import Pool, cpu_count

from ctc_metrics.metrics import valid, det, seg, tra, ct, tf, bc, cca, \
    det_original, seg_original, tra_original
from ctc_metrics.utils.handle_results import print_results, store_results
from ctc_metrics.utils.filesystem import parse_directories, read_tracking_file,\
    parse_masks
from ctc_metrics.utils.representations import match as match_tracks


def evaluate_sequence(
        res: str,
        gt: str,
        num_digits: int,
        st: str = None,
        metrics: list = None,
        multiprocessing: bool = True,
    ):
    """ Evaluate a single sequence """
    print("\r", res, end="")
    if metrics is None:
        """ Verify all metrics """
        metrics = ["Valid", "DET", "SEG", "TRA", "CT", "TF", "BC", "CCA"]
    res_tracks = read_tracking_file(join(res, "res_track.txt"))
    gt_tracks = read_tracking_file(join(gt, "TRA", "man_track.txt"))
    res_masks = parse_masks(res)
    gt_tra_masks = parse_masks(join(gt, "TRA"))
    gt_seg_masks = parse_masks(join(gt, "SEG"))
    if st is not None:
        st_seg_masks = parse_masks(join(st, "SEG"))
    assert len(gt_tra_masks) > 0, res
    assert len(gt_tra_masks) == len(res_masks)

    # Parse golden truth tracking masks
    args = zip(gt_tra_masks, res_masks)
    if multiprocessing:
        with Pool(cpu_count()) as p:
            matches = p.starmap(match_tracks, args)
    else:
        matches = [match_tracks(*x) for x in args]
    labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra, ious_tra = \
        [], [], [], [], []
    for match in matches:
        labels_gt_tra.append(match[0])
        labels_res_tra.append(match[1])
        mapped_gt_tra.append(match[2])
        mapped_res_tra.append(match[3])
        ious_tra.append(match[4])

    # Parse golden truth segmentation masks
    _res_masks = [
        res_masks[int(basename(x).replace("man_seg", "").replace(".tif", ""))]
        for x in gt_seg_masks
    ]
    args = zip(gt_seg_masks, _res_masks)
    if multiprocessing:
        with Pool(cpu_count()) as p:
            matches = p.starmap(match_tracks, args)
    else:
        matches = [match_tracks(*x) for x in args]
    labels_gt_seg, labels_res_seg, mapped_gt_seg, mapped_res_seg, ious_seg = \
        [], [], [], [], []
    for match in matches:
        labels_gt_seg.append(match[0])
        labels_res_seg.append(match[1])
        mapped_gt_seg.append(match[2])
        mapped_res_seg.append(match[3])
        ious_seg.append(match[4])

    results = dict()
    if "Valid" in metrics:
        results["Valid"] = valid(res_masks, res_tracks, labels_res_tra)
    if "DET" in metrics:
        results["DET_ORIGINAL"] = det_original(res, num_digits)
        DET, TP, FN, FP, NS = det(
            labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra)
        results["DET"] = DET
        results["DET_TP"] = TP
        results["DET_FN"] = FN
        results["DET_FP"] = FP
        results["DET_NS"] = NS
    if "SEG" in metrics:
        results["SEG_ORIGINAL"] = seg_original(res, num_digits)
        seg_measurement, true_positives, false_negatives = seg(
            labels_gt_seg, ious_seg)
        results["SEG"] = seg_measurement
        results["SEG_TP"] = true_positives
        results["SEG_FN"] = false_negatives
    if "TRA" in metrics:
        results["TRA_ORIGINAL"] = tra_original(res, num_digits)
        TRA, NS, FN, FP, ED, EA, EC = tra(
            gt_tracks, res_tracks,
            labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra
        )
        results["TRA"] = TRA
        results["TRA_NS"] = NS
        results["TRA_FN"] = FN
        results["TRA_FP"] = FP
        results["TRA_ED"] = ED
        results["TRA_EA"] = EA
        results["TRA_EC"] = EC
    if "CT" in metrics:
        results["CT"] = ct(
            res_tracks, gt_tracks, labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra
        )
    if "TF" in metrics:
        results["TF"] = tf(
            res_tracks, gt_tracks, labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra
        )
    if "BC" in metrics:
        results["BC(0)"] = bc(
            res_tracks, gt_tracks, labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra,
            i=0
        )
        results["BC(1)"] = bc(
            res_tracks, gt_tracks, labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra,
            i=1
        )
        results["BC(2)"] = bc(
            res_tracks, gt_tracks, labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra,
            i=2
        )
        results["BC(3)"] = bc(
            res_tracks, gt_tracks, labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra,
            i=3
        )
    if "CCA" in metrics:
        results["CCA"] = cca(res_tracks, gt_tracks)
    print("\r", end="")

    return results


def evaluate_all(
        res_root: str,
        gt_root: str,
        metrics: list = None,
    ):
    """ Evaluate all sequences in a directory """
    results = list()

    ret = parse_directories(res_root, gt_root)
    for res, gt, st, num_digits, name in zip(*ret):
        results.append(
            [name, evaluate_sequence(res, gt, st, num_digits, metrics)]
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluates CTC-Sequences. '
    )

    parser.add_argument('--res', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--num-digits', type=int, default=None)
    parser.add_argument('--full-directory', action="store_true")
    parser.add_argument('--csv-path', type=str, default=None)
    parser.add_argument('--Valid', action="store_true")
    parser.add_argument('--DET', action="store_true")
    parser.add_argument('--SEG', action="store_true")
    parser.add_argument('--TRA', action="store_true")
    parser.add_argument('--CT', action="store_true")
    parser.add_argument('--TF', action="store_true")
    parser.add_argument('--BC', action="store_true")
    parser.add_argument('--CCA', action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    metrics = list()
    if args.Valid:
        metrics.append("Valid")
    if args.DET:
        metrics.append("DET")
    if args.SEG:
        metrics.append("SEG")
    if args.TRA:
        metrics.append("TRA")
    if args.CT:
        metrics.append("CT")
    if args.TF:
        metrics.append("TF")
    if args.BC:
        metrics.append("BC")
    if args.CCA:
        metrics.append("CCA")
    if len(metrics) == 0:
        metrics = None

    if args.full_directory:
        res = evaluate_all(
            res_root=args.res,
            gt_root=args.gt,
            metrics=metrics
        )
    else:
        res = evaluate_sequence(
            res=args.res,
            gt=args.gt,
            metrics=metrics,
            num_digits=args.num_digits,
        )

    print_results(res)
    if args.csv_path is not None:
        store_results(args.csv_path, res)


# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train" --csv-path="C:\Users\kaiser\Desktop\data\CTC\Inference\original\eval.csv" --full-directory
# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train" --csv-path="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\eval.csv" --full-directory
# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_20_5_no_mitosis\train" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_20_5_no_mitosis\train" --csv-path="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_20_5_no_mitosis\eval.csv" --full-directory


# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train\BF-C2DL-HSC\01_RES" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train\BF-C2DL-HSC\01_GT" --Valid
# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train\BF-C2DL-HSC\01_RES" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train\BF-C2DL-HSC\01_GT" --Valid
