import argparse

from os.path import join, basename, exists
from multiprocessing import Pool, cpu_count

from ctc_metrics.metrics import valid, det, seg, tra, ct, tf, bc, cca
from ctc_metrics.utils.handle_results import print_results, store_results
from ctc_metrics.utils.filesystem import parse_directories, read_tracking_file,\
    parse_masks
from ctc_metrics.utils.representations import match as match_tracks, \
    count_acyclic_graph_correction_operations


def match_computed_to_reference_masks(
        ref_masks: list,
        comp_masks: list,
        threads: int = 0,
):
    """
    Matches computed masks to reference masks.

    Args:
        ref_masks: The reference masks.
        comp_masks: The computed masks.
        threads: The number of threads to use. If 0, the number of threads
            is set to the number of available CPUs.

    Returns:
        The results stored in a dictionary.
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


def evaluate_sequence(
        res: str,
        gt: str,
        metrics: list = None,
        threads: int = 0,
    ):  # pylint: disable=too-complex
    """
    Evaluate a single sequence

    Args:
        res: The path to the results.
        gt: The path to the ground truth.
        metrics: The metrics to evaluate.
        multiprocessing: Whether to use multiprocessing (recommended!).
        threads: The number of threads to use. If 0, the number of threads
            is set to the number of available CPUs.

    Returns:
        The results stored in a dictionary.
    """
    print("\r", res, end=": ")
    # Verify all metrics
    if metrics is None:
        metrics = ["Valid", "DET", "SEG", "TRA", "CT", "TF", "BC", "CCA"]
    # Read tracking files and parse mask files
    comp_tracks = read_tracking_file(join(res, "res_track.txt"))
    ref_tracks = read_tracking_file(join(gt, "TRA", "man_track.txt"))
    ref_tra_masks = parse_masks(join(gt, "TRA"))
    assert len(ref_tra_masks) > 0, f"{res}: Ground truth masks is 0!)"
    # Match golden truth tracking masks to result masks
    traj = {}
    is_valid = None
    if sorted(metrics) != ["CCA"]:
        comp_masks = [
            join(res, "mask" + basename(x).replace(
                "man_track", "").replace(".tif", "").replace("_", "") + ".tif")
            for x in ref_tra_masks
        ]
        comp_masks = [x if exists(x) else None for x in comp_masks]
        traj = match_computed_to_reference_masks(
            ref_tra_masks, comp_masks, threads=threads)
        is_valid = valid(comp_masks, comp_tracks, traj["labels_comp"])
    # Match golden truth segmentation masks to result masks
    segm = {}
    if "SEG" in metrics:
        ref_seg_masks = parse_masks(join(gt, "SEG"))
        comp_masks = [
            join(res, "mask" + basename(x).replace(
                "man_seg", "").replace(".tif", "").replace("_", "") + ".tif")
            for x in ref_seg_masks
        ]
        comp_masks = [x if exists(x) else None for x in comp_masks]
        segm = match_computed_to_reference_masks(
            ref_seg_masks, comp_masks, threads=threads)
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
    results = {}
    if "Valid" in metrics:
        results["Valid"] = is_valid
    if "DET" in metrics:
        results["DET"] = det(**graph_operations)
    if "SEG" in metrics:
        results["SEG"] = seg(segm["labels_ref"], segm["ious"])
    if "TRA" in metrics:
        results["TRA"] = tra(**graph_operations)
    if "CT" in metrics:
        results["CT"] = ct(
            comp_tracks, ref_tracks,
            traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])
    if "TF" in metrics:
        results["TF"] = tf(
            ref_tracks,
            traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])
    if "BC" in metrics:
        for i in range(6):
            results[f"BC({i})"] = bc(
                comp_tracks, ref_tracks,
                traj["mapped_ref"], traj["mapped_comp"],
                i=i)
    if "CCA" in metrics:
        results["CCA"] = cca(comp_tracks, ref_tracks)
    print(results)
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
        ("CCA", args.cca)
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
