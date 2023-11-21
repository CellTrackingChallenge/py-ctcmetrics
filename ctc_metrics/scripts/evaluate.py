import argparse
from os.path import join, basename
from multiprocessing import Pool, cpu_count

from ctc_metrics.metrics import valid, det, seg, tra, ct, tf, bc, cca
from ctc_metrics.utils.handle_results import print_results, store_results
from ctc_metrics.utils.filesystem import parse_directories, read_tracking_file,\
    parse_masks
from ctc_metrics.utils.representations import match as match_tracks, \
    merge_tracks,count_acyclic_graph_correction_operations


def evaluate_sequence(
        res: str,
        gt: str,
        metrics: list = None,
        multiprocessing: bool = True,
    ):
    """
    Evaluate a single sequence

    Args:
        res: The path to the results.
        gt: The path to the ground truth.
        metrics: The metrics to evaluate.
        multiprocessing: Whether to use multiprocessing (recommended!).

    Returns:
        The results stored in a dictionary.
    """
    print("\r", res, end="")
    if metrics is None:
        """ Verify all metrics """
        metrics = ["Valid", "DET", "SEG", "TRA", "CT", "TF", "BC", "CCA"]
    # Read tracking files and parse mask files
    res_tracks = read_tracking_file(join(res, "res_track.txt"))
    gt_tracks = read_tracking_file(join(gt, "TRA", "man_track.txt"))
    res_masks = parse_masks(res)
    gt_tra_masks = parse_masks(join(gt, "TRA"))
    gt_seg_masks = parse_masks(join(gt, "SEG"))
    assert len(gt_tra_masks) > 0, res
    assert len(gt_tra_masks) == len(res_masks)

    # Match golden truth tracking masks to result masks
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

    # Match golden truth segmentation masks to result masks
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

    # Prepare intermediate results
    NS, FN, FP, ED, EA, EC, num_vertices, num_edges = \
        count_acyclic_graph_correction_operations(
            gt_tracks, res_tracks,
            labels_gt_tra, labels_res_tra, mapped_gt_tra, mapped_res_tra
        )
    original_tracks = [
        res_tracks, gt_tracks, labels_gt_tra, labels_res_tra, mapped_gt_tra,
        mapped_res_tra
    ]
    merged_tracks = merge_tracks(*original_tracks)

    # Calculate metrics
    results = dict()

    if "Valid" in metrics:
        results["Valid"] = valid(res_masks, res_tracks, labels_res_tra)
    if "DET" in metrics:
        results["DET"] = det(NS, FN, FP, num_vertices)
        results["AOGM_NS"] = NS
        results["AOGM_FN"] = FN
        results["AOGM_FP"] = FP
    if "SEG" in metrics:
        SEG, SEG_TP, SEG_FN = seg(labels_gt_seg, ious_seg)
        results["SEG"] = SEG
        results["SEG_TP"] = SEG_TP
        results["SEG_FN"] = SEG_FN
    if "TRA" in metrics:
        results["TRA"] = tra(NS, FN, FP, ED, EA, EC, num_vertices, num_edges)
        results["AOGM_ED"] = ED
        results["AOGM_EA"] = EA
        results["AOGM_EC"] = EC
    if "CT" in metrics:
        results["CT"] = ct(*merged_tracks)
    if "TF" in metrics:
        results["TF"] = tf(*merged_tracks)
    if "BC" in metrics:
        results["BC(0)"] = bc(*merged_tracks, i=0)
        results["BC(1)"] = bc(*merged_tracks, i=1)
        results["BC(2)"] = bc(*merged_tracks, i=2)
        results["BC(3)"] = bc(*merged_tracks, i=3)
    if "CCA" in metrics:
        results["CCA"] = cca(*merged_tracks)
    print("\r", end="")

    return results


def evaluate_all(
        res_root: str,
        gt_root: str,
        metrics: list = None,
    ):
    """
    Evaluate all sequences in a directory

    Args:
        res_root: The root directory of the results.
        gt_root: The root directory of the ground truth.
        metrics: The metrics to evaluate.

    Returns:
        The results stored in a dictionary.
    """
    results = list()

    ret = parse_directories(res_root, gt_root)
    for res, gt, name in zip(*ret):
        results.append(
            [name, evaluate_sequence(res, gt, metrics)]
        )

    return results


def parse_args():
    """ Parse arguments """
    # todo: Add help comments to all arguments
    parser = argparse.ArgumentParser(
        description='Evaluates CTC-Sequences. '
    )

    parser.add_argument('--res', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('-r', '--recursive', action="store_true")
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


def main():
    """
    Main function that is called when the script is executed.
    """
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

    if args.recursive:
        res = evaluate_all(res_root=args.res, gt_root=args.gt, metrics=metrics)
    else:
        res = evaluate_sequence(res=args.res, gt=args.gt, metrics=metrics)

    print_results(res)
    if args.csv_path is not None:
        store_results(args.csv_path, res)


if __name__ == "__main__":
    main()
