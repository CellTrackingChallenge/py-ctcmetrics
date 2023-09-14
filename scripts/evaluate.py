import argparse
from os.path import join, basename, dirname
from multiprocessing import Pool, cpu_count

from metrics import valid, det, seg, tra, ct, tf, bc, cca
from utils.handle_results import print_results, store_results
from utils.filesystem import parse_directories, read_tracking_file, parse_masks
from utils.representations import match as match_tracks, cluster_full_tracks


def evaluate_sequence(
        res: str,
        gt: str,
        num_digits: int,
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
    gt_masks = parse_masks(join(gt, "TRA"))
    assert len(gt_masks) > 0, res
    assert len(gt_masks) == len(res_masks)

    args = zip(gt_masks, res_masks)
    if multiprocessing:
        with Pool(cpu_count()) as p:
            matches = p.starmap(match_tracks, args)
    else:
        matches = [match_tracks(*x) for x in args]
    labels_gt, labels_res, mapped_gt, mapped_res = [], [], [], []
    for match in matches:
        labels_gt.append(match[0])
        labels_res.append(match[1])
        mapped_gt.append(match[2])
        mapped_res.append(match[3])

    results = dict()
    if "Valid" in metrics:
        results["Valid"] = valid(res_masks, res_tracks, labels_res)
    if "DET" in metrics:
        results["DET"] = det(res, num_digits)
    if "SEG" in metrics:
        results["SEG"] = seg(res, num_digits)
    if "TRA" in metrics:
        results["TRA"] = tra(res, num_digits)
    if "CT" in metrics:
        results["CT"] = ct(
            res_tracks, gt_tracks, labels_gt, labels_res, mapped_gt, mapped_res
        )
    if "TF" in metrics:
        results["TF"] = tf(
            res_tracks, gt_tracks, labels_gt, labels_res, mapped_gt, mapped_res
        )
    if "BC" in metrics:
        results["BC(0)"] = bc(
            res_tracks, gt_tracks, labels_gt, labels_res, mapped_gt, mapped_res,
            i=0
        )
        results["BC(1)"] = bc(
            res_tracks, gt_tracks, labels_gt, labels_res, mapped_gt, mapped_res,
            i=1
        )
        results["BC(2)"] = bc(
            res_tracks, gt_tracks, labels_gt, labels_res, mapped_gt, mapped_res,
            i=2
        )
        results["BC(3)"] = bc(
            res_tracks, gt_tracks, labels_gt, labels_res, mapped_gt, mapped_res,
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
    for res, gt, num_digits, name in zip(*ret):
        results.append([name, evaluate_sequence(res, gt, num_digits, metrics)])

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
        res = evaluate_all(args.res, args.gt, metrics)
    else:
        res = evaluate_sequence(args.res, args.gt, args.num_digits, metrics)

    print_results(res)
    if args.csv_path is not None:
        store_results(args.csv_path, res)


# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train" --csv-path="C:\Users\kaiser\Desktop\data\CTC\Inference\original\eval.csv" --full-directory
# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train" --csv-path="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\eval.csv" --full-directory
# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_20_5_no_mitosis\train" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_20_5_no_mitosis\train" --csv-path="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_20_5_no_mitosis\eval.csv" --full-directory


# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train\BF-C2DL-HSC\01_RES" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\original\train\BF-C2DL-HSC\01_GT" --Valid
# python scripts/evaluate.py --res="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train\BF-C2DL-HSC\01_RES" --gt="C:\Users\kaiser\Desktop\data\CTC\Inference\ours_1_1_no_mitosis\train\BF-C2DL-HSC\01_GT" --Valid
