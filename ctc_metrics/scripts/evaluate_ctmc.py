import argparse
import os.path
from os.path import join
from ctc_metrics.metrics import valid
from ctc_metrics.metrics import ALL_METRICS
from ctc_metrics.utils.handle_results import print_results, store_results
from ctc_metrics.utils.filesystem import parse_directories, read_tracking_file, load_ctmc_bounding_boxes
from ctc_metrics.utils.representations import match_bboxes
from ctc_metrics.scripts.evaluate import calculate_metrics

def match_computed_to_reference_masks(
        ref_boxes: list,
        comp_boxes: list,
):
    """
    Matches computed masks to reference masks.

    Args:
        ref_boxes: The reference masks. A list of lists of the reference bboxes. [frame, id, x, y, w, h]
        comp_boxes: The computed masks. A list of lists of the computed bboxes. [frame, id, x, y, w, h]


    Returns:
        The results stored in a dictionary. The dictionary contains the
        following keys:
            - labels_ref: The reference labels. A list of lists containing
                the labels of the reference masks.
            - labels_comp: The computed labels. A list of lists containing
                the labels of the computed masks.
            - mapped_ref: The mapped reference labels. A list of lists
                containing the mapped labels of the reference masks.
            - mapped_comp: The mapped computed labels. A list of lists
                containing the mapped labels of the computed masks.
            - ious: The intersection over union values. A list of lists
                containing the intersection over union values between mapped
                reference and computed masks.
    """
    labels_ref, labels_comp, mapped_ref, mapped_comp, ious = [], [], [], [], []

    matches = [match_bboxes(*x) for x in zip(ref_boxes, comp_boxes)]
    for match in matches:
        labels_ref.append(match[0])
        labels_comp.append(match[1])
        mapped_ref.append(match[2])
        mapped_comp.append(match[3])
        ious.append(match[4])
    return {
        "labels_ref": labels_ref,
        "labels_comp": labels_comp,
        "mapped_ref": mapped_ref,
        "mapped_comp": mapped_comp,
        "ious": ious
    }


def load_data(
        res: str,
        gt: str,
):
    """
    Load data that is necessary to calculate metrics from the given directories.

    Args:
        res: The path to the results.
        gt: The path to the ground truth.
        trajectory_data: A flag if trajectory data is available.
        segmentation_data: A flag if segmentation data is available.
        threads: The number of threads to use. If 0, the number of threads
            is set to the number of available CPUs.

    Returns:
        The computed tracks, the reference tracks, the trajectory data, the
        segmentation data, the computed masks and a flag if the results are
        valid.

    """
    # Read tracking files and parse mask files
    comp_tracking_file = join(res, "TRA", "res_track.txt")
    assert os.path.exists(comp_tracking_file), f"{comp_tracking_file} does not exist."
    comp_tracks = read_tracking_file(comp_tracking_file)
    ref_tracking_file = join(gt, "TRA", "man_track.txt")
    assert os.path.exists(ref_tracking_file), f"{ref_tracking_file} does not exist."
    ref_tracks = read_tracking_file(ref_tracking_file)
    comp_bb_file = join(res, "res", "res.txt")
    assert os.path.exists(comp_bb_file), f"{comp_bb_file} does not exist."
    comp_masks = load_ctmc_bounding_boxes(comp_bb_file)
    ref_bb_file = join(gt, "gt", "gt.txt")
    assert os.path.exists(ref_bb_file), f"{ref_bb_file} does not exist."
    ref_tra_masks = load_ctmc_bounding_boxes(ref_bb_file)
    assert len(ref_tra_masks) > 0, f"{gt}: Ground truth masks is 0!)"
    assert len(ref_tra_masks) == len(comp_masks), (
        f"{res}: Number of result masks ({len(comp_masks)}) unequal to "
        f"the number of ground truth masks ({len(ref_tra_masks)})!)")
    # Match golden truth tracking masks to result masks
    traj = match_computed_to_reference_masks(ref_tra_masks, comp_masks)
    is_valid = valid(None, comp_tracks, traj["labels_comp"])
    # Match golden truth segmentation masks to result masks
    return comp_tracks, ref_tracks, traj, comp_masks, is_valid


def evaluate_sequence(
        res: str,
        gt: str,
        metrics: list = None,
    ):
    """
    Evaluates a single sequence.

    Args:
        res: The path to the results.
        gt: The path to the ground truth.
        metrics: The metrics to evaluate.
        threads: The number of threads to use. If 0, the number of threads
            is set to the number of available CPUs.

    Returns:
        The results stored in a dictionary.
    """

    print("Evaluate sequence: ", res, " with ground truth: ", gt, end="")
    # Verify all metrics
    if metrics is None:
        metrics = ALL_METRICS
        if "SEG" in metrics:
            metrics.remove("SEG")  # SEG is not existing for CTMC


    comp_tracks, ref_tracks, traj,  _, is_valid = load_data(res, gt)

    results = calculate_metrics(
        comp_tracks, ref_tracks, traj, {}, metrics, is_valid)

    print("with results: ", results, " done!")

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
        threads: The number of threads to use. If 0, the number of threads
            is set to the number of available CPUs.

    Returns:
        The results stored in a dictionary.
    """
    results = []
    ret = parse_directories(res_root, gt_root)
    for res, gt, name in zip(*ret):
        results.append([name, evaluate_sequence(res, gt, metrics)])
    return results


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Evaluates CTC-Sequences.')
    parser.add_argument('--res', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('-r', '--recursive', action="store_true")
    parser.add_argument('--csv-file', type=str, default=None)
    parser.add_argument('-n', '--num-threads', type=int, default=0)
    parser.add_argument('--valid', action="store_true")
    parser.add_argument('--det', action="store_true")
    parser.add_argument('--seg', action="store_true")
    parser.add_argument('--tra', action="store_true")
    parser.add_argument('--ct', action="store_true")
    parser.add_argument('--tf', action="store_true")
    parser.add_argument('--bc', action="store_true")
    parser.add_argument('--cca', action="store_true")
    parser.add_argument('--mota', action="store_true")
    parser.add_argument('--hota', action="store_true")
    parser.add_argument('--idf1', action="store_true")
    parser.add_argument('--chota', action="store_true")
    parser.add_argument('--mtml', action="store_true")
    parser.add_argument('--faf', action="store_true")
    parser.add_argument('--lnk', action="store_true")
    args = parser.parse_args()
    return args


def main():
    """
    Main function that is called when the script is executed.
    """
    args = parse_args()
    # Prepare metric selection
    metrics = [metric for metric, flag in (
        ("Valid", args.valid),
        ("DET", args.det),
        ("SEG", False),
        ("TRA", args.tra),
        ("CT", args.ct),
        ("TF", args.tf),
        ("BC", args.bc),
        ("CCA", args.cca),
        ("MOTA", args.mota),
        ("HOTA", args.hota),
        ("CHOTA", args.chota),
        ("IDF1", args.idf1),
        ("MTML", args.mtml),
        ("FAF", args.faf),
        ("LNK", args.lnk),
    ) if flag]
    metrics = metrics if metrics else None
    # Evaluate sequence or whole directory
    if args.recursive:
        res = evaluate_all(
            res_root=args.res, gt_root=args.gt, metrics=metrics,
        )
    else:
        res = evaluate_sequence(
            res=args.res, gt=args.gt, metrics=metrics)
    # Visualize and store results
    print_results(res)
    if args.csv_file is not None:
        store_results(args.csv_file, res)


if __name__ == "__main__":
    main()
