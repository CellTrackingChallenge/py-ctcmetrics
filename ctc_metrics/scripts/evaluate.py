import argparse
from os.path import join, basename
from multiprocessing import Pool, cpu_count
import numpy as np

from ctc_metrics.metrics import (
    valid, det, seg, tra, ct, tf, bc, cca, mota, hota, idf1, chota, mtml, faf,
    op_ctb, op_csb, bio, op_clb, lnk
)
from ctc_metrics.metrics import ALL_METRICS
from ctc_metrics.utils.handle_results import print_results, store_results
from ctc_metrics.utils.filesystem import parse_directories, read_tracking_file,\
    parse_masks
from ctc_metrics.utils.representations import match as match_tracks, \
    count_acyclic_graph_correction_operations, merge_tracks


def match_computed_to_reference_masks(
        ref_masks: list,
        comp_masks: list,
        threads: int = 0,
):
    """
    Matches computed masks to reference masks.

    Args:
        ref_masks: The reference masks. A list of paths to the reference masks.
        comp_masks: The computed masks. A list of paths to the computed masks.
        threads: The number of threads to use. If 0, the number of threads
            is set to the number of available CPUs.

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
    if threads != 1:
        if threads == 0:
            threads = cpu_count()
        with Pool(threads) as p:
            matches = p.starmap(match_tracks, zip(ref_masks, comp_masks))
    else:
        matches = [match_tracks(*x) for x in zip(ref_masks, comp_masks)]
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
        trajectory_data: True,
        segmentation_data: True,
        threads: int = 0,
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
    comp_tracks = read_tracking_file(join(res, "res_track.txt"))
    ref_tracks = read_tracking_file(join(gt, "TRA", "man_track.txt"))
    comp_masks = parse_masks(res)
    ref_tra_masks = parse_masks(join(gt, "TRA"))
    assert len(ref_tra_masks) > 0, f"{gt}: Ground truth masks is 0!)"
    assert len(ref_tra_masks) == len(comp_masks), (
        f"{res}: Number of result masks ({len(comp_masks)}) unequal to "
        f"the number of ground truth masks ({len(ref_tra_masks)})!)")
    # Match golden truth tracking masks to result masks
    traj = {}
    is_valid = 1
    if trajectory_data:
        traj = match_computed_to_reference_masks(
            ref_tra_masks, comp_masks, threads=threads)
        is_valid = valid(comp_masks, comp_tracks, traj["labels_comp"])
    # Match golden truth segmentation masks to result masks
    segm = {}
    if segmentation_data:
        ref_seg_masks = parse_masks(join(gt, "SEG"))
        _res_masks = [
            comp_masks[int(basename(x).replace(
                "man_seg", "").replace(".tif", "").replace("_", ""))]
            for x in ref_seg_masks
        ]
        segm = match_computed_to_reference_masks(
            ref_seg_masks, _res_masks, threads=threads)
    return comp_tracks, ref_tracks, traj, segm, comp_masks, is_valid


def calculate_metrics(
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray,
        traj: dict,
        segm: dict,
        metrics: list = None,
        is_valid: bool = None,
):  # noqa: C901
    """
    Calculate metrics for given data.

    Args:
        comp_tracks: The computed tracks.A (n,4) numpy ndarray with columns:
            - label
            - birth frame
            - end frame
            - parent
        ref_tracks: The reference tracks. A (n,4) numpy ndarray with columns:
            - label
            - birth frame
            - end frame
            - parent
        traj: The frame-wise trajectory match data.
        segm: The frame-wise segmentation match data.
        metrics: The metrics to evaluate.
        is_valid: A Flag if the results are valid

    Returns:
        The results stored in a dictionary.
    """
    # Create merge tracks
    if traj:
        new_tracks, new_labels, new_mapped = merge_tracks(
            ref_tracks, traj["labels_ref"], traj["mapped_ref"])
        traj["ref_tracks_merged"] = new_tracks
        traj["labels_ref_merged"] = new_labels
        traj["mapped_ref_merged"] = new_mapped
        new_tracks, new_labels, new_mapped = merge_tracks(
            comp_tracks, traj["labels_comp"], traj["mapped_comp"])
        traj["comp_tracks_merged"] = new_tracks
        traj["labels_comp_merged"] = new_labels
        traj["mapped_comp_merged"] = new_mapped

    # Prepare intermediate results
    graph_operations = {}
    if "DET" in metrics or "TRA" in metrics:
        graph_operations = \
            count_acyclic_graph_correction_operations(
                ref_tracks, comp_tracks,
                traj["labels_ref"], traj["labels_comp"],
                traj["mapped_ref"], traj["mapped_comp"]
            )

    # Calculate metrics
    results = {x: None for x in metrics}
    if not is_valid:
        print("Invalid results!")
        results["Valid"] = 0
        return results

    if "Valid" in metrics:
        results["Valid"] = is_valid

    if "CHOTA" in metrics:
        results.update(chota(
            traj["ref_tracks_merged"], traj["comp_tracks_merged"],
            traj["labels_ref_merged"], traj["labels_comp_merged"],
            traj["mapped_ref_merged"], traj["mapped_comp_merged"]))

    if "DET" in metrics:
        results["DET"] = det(**graph_operations)

    if "SEG" in metrics:
        results["SEG"] = seg(segm["labels_ref"], segm["ious"])

    if "TRA" in metrics:
        results["TRA"] = tra(**graph_operations)

    if "LNK" in metrics:
        results["LNK"] = lnk(**graph_operations)

    if "DET" in metrics and "SEG" in metrics:
        results["OP_CSB"] = op_csb(results["SEG"], results["DET"])

    if "SEG" in metrics and "TRA" in metrics:
        results["OP_CTB"] = op_ctb(results["SEG"], results["TRA"])

    if "CT" in metrics:
        results["CT"] = ct(
            comp_tracks, ref_tracks,
            traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])

    if "TF" in metrics:
        results["TF"] = tf(
            ref_tracks,
            traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])

    if "BC" in metrics:
        for i in range(4):
            results[f"BC({i})"] = bc(
                comp_tracks, ref_tracks,
                traj["mapped_ref"], traj["mapped_comp"],
                i=i)

    if "CCA" in metrics:
        results["CCA"] = cca(comp_tracks, ref_tracks)

    if "CT" in metrics and "BC" in metrics and \
            "CCA" in metrics and "TF" in metrics:
        for i in range(4):
            results[f"BIO({i})"] = bio(
                results["CT"], results["TF"],
                results[f"BC({i})"], results["CCA"])

    if "BIO" in results and "LNK" in results:
        for i in range(4):
            results[f"OP_CLB({i})"] = op_clb(
                results["LNK"], results[f"BIO({i})"])

    if "MOTA" in metrics:
        results.update(mota(
            traj["labels_ref_merged"], traj["labels_comp_merged"],
            traj["mapped_ref_merged"], traj["mapped_comp_merged"]))

    if "HOTA" in metrics:
        results.update(hota(
            traj["labels_ref_merged"], traj["labels_comp_merged"],
            traj["mapped_ref_merged"], traj["mapped_comp_merged"]))

    if "IDF1" in metrics:
        results.update(idf1(
            traj["labels_ref_merged"], traj["labels_comp_merged"],
            traj["mapped_ref_merged"], traj["mapped_comp_merged"]))

    if "MTML" in metrics:
        results.update(mtml(
            traj["labels_ref_merged"], traj["labels_comp_merged"],
            traj["mapped_ref_merged"], traj["mapped_comp_merged"]))

    if "FAF" in metrics:
        results.update(faf(
            traj["labels_comp_merged"], traj["mapped_comp_merged"]))

    return results


def evaluate_sequence(
        res: str,
        gt: str,
        metrics: list = None,
        threads: int = 0,
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

    trajectory_data = True
    segmentation_data = True

    if metrics in [["SEG"], ["CCA"]]:
        trajectory_data = False

    if "SEG" not in metrics:
        segmentation_data = False

    comp_tracks, ref_tracks, traj, segm, _, is_valid = load_data(
        res, gt, trajectory_data, segmentation_data, threads)

    results = calculate_metrics(
        comp_tracks, ref_tracks, traj, segm, metrics, is_valid)

    print("with results: ", results, " done!")

    return results


def evaluate_all(
        res_root: str,
        gt_root: str,
        metrics: list = None,
        threads: int = 0
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
        results.append([name, evaluate_sequence(res, gt, metrics, threads)])
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
        ("SEG", args.seg),
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
            threads=args.num_threads
        )
    else:
        res = evaluate_sequence(
            res=args.res, gt=args.gt, metrics=metrics, threads=args.num_threads)
    # Visualize and store results
    print_results(res)
    if args.csv_file is not None:
        store_results(args.csv_file, res)


if __name__ == "__main__":
    main()
