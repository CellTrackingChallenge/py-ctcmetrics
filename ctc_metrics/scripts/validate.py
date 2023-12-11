import argparse
from os.path import join
from multiprocessing import Pool, cpu_count

from ctc_metrics.metrics import valid
from ctc_metrics.utils.handle_results import print_results
from ctc_metrics.utils.filesystem import \
    parse_directories, read_tracking_file, parse_masks
from ctc_metrics.utils.representations import match as match_tracks


def validate_sequence(
        res: str,
        multiprocessing: bool = True,
):
    """
    Validates a single sequence

    Args:
        res: The path to the results.
        multiprocessing: Whether to use multiprocessing (recommended!).

    Returns:
        The results stored in a dictionary.
    """
    print("\r", res, end="")
    res_tracks = read_tracking_file(join(res, "res_track.txt"))
    res_masks = parse_masks(res)
    assert len(res_masks) > 0, res
    args = zip([None for x in res_masks], res_masks)
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
    results = {"Valid": valid(res_masks, res_tracks, labels_res)}
    print("\r", end="")
    return results


def validate_all(
        res_root: str,
):
    """
    Evaluate all sequences in a directory

    Args:
        res_root: The path to the result directory.

    Returns:
        The results stored in a dictionary.
    """
    results = []
    ret = parse_directories(res_root, None)
    for res, gt, name in zip(*ret):
        results.append([name, validate_sequence(res, gt)])
    return results


def parse_args():
    """ Parses the arguments. """
    parser = argparse.ArgumentParser(description='Validates CTC-Sequences.')
    parser.add_argument('--res', type=str, required=True)
    parser.add_argument('-r', '--recursive', action="store_true")
    args = parser.parse_args()
    return args


def main():
    """
    Main function that is called when the script is executed.
    """
    args = parse_args()
    if args.recursive:
        res = validate_all(args.res)
    else:
        res = validate_sequence(args.res)
    print_results(res)


if __name__ == "__main__":
    main()

