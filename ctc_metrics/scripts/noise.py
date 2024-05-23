import argparse
from os.path import join, basename, dirname, exists
import copy
import numpy as np
import pandas as pd
import random

from ctc_metrics.metrics import ALL_METRICS
from ctc_metrics.utils.filesystem import parse_directories, read_tracking_file, \
    parse_masks
from ctc_metrics.utils.representations import merge_tracks

from ctc_metrics.scripts.evaluate import match_computed_to_reference_masks, \
    calculate_metrics


def load_data(
        gt: str,
        threads: int = 0,
):
    # Read tracking files and parse mask files
    ref_tracks = read_tracking_file(join(gt, "TRA", "man_track.txt"))
    comp_tracks = np.copy(ref_tracks)
    ref_tra_masks = parse_masks(join(gt, "TRA"))
    comp_masks = ref_tra_masks
    assert len(ref_tra_masks) > 0, f"{gt}: Ground truth masks is 0!)"

    # Match golden truth tracking masks to result masks
    traj = match_computed_to_reference_masks(
        ref_tra_masks, comp_masks, threads=threads)

    # Match golden truth segmentation masks to result masks
    segm = {}

    return comp_tracks, ref_tracks, traj, segm, comp_masks


def add_noise(
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray,
        traj: dict,
        seed: int = 0,
        noise_add_false_negative: int = 0,
        noise_add_false_positive: int = 0,
        noise_add_idsw: int = 0,
        noise_remove_matches: int = 0,
        noise_remove_mitosis: int = 0,
):
    """
    Add noise to the data.

    Args:
        comp_tracks:
        ref_tracks:
        traj:
        seed:
        noise_add_false_negative:
            Adds n false negatives to the data, where the parameter is n.
        noise_remove_mitosis:
            Removes parend daughter relations of n mitosis events, where the
            parameter describes n.
        noise_add_false_positive:
            Adds n false positives to the data, where the parameter is n.
        noise_add_idsw:
            Adds n ID switches to the data, where the parameter is n.
        noise_remove_matches:
            Removes n matches from the data, where the parameter is n.
            This produces n false negatives and n false positives.

    Returns:
        comp_tracks: Updated with noise applied.
        traj: Updated with noise applied.
    """
    comp_tracks = np.copy(comp_tracks)
    traj = copy.deepcopy(traj)

    # Remove all children of mitosis events
    if noise_remove_mitosis > 0:
        parents, counts = np.unique(
            comp_tracks[comp_tracks[:, 3] > 0, 3], return_counts=True)
        parents = parents[counts > 1]
        num_splits = min(noise_remove_mitosis, len(parents))
        np.random.seed(seed)
        np.random.shuffle(parents)
        for parent in parents[:num_splits]:
            comp_tracks[np.isin(comp_tracks[:, 3], parent), 3] = 0

    # Add false negatives
    if noise_add_false_negative > 0:
        next_id = np.max(comp_tracks[:, 0]) + 1
        l_comp = traj["labels_comp"]
        m_comp = traj["mapped_comp"]
        m_ref = traj["mapped_ref"]
        candidates = []
        for frame in range(0, len(l_comp)):
            for i in range(len(l_comp[frame])):
                candidates.append((frame, i))
        np.random.seed(seed)
        np.random.shuffle(candidates)
        num_fn = min(noise_add_false_negative, len(candidates))
        for frame, i in candidates[:num_fn]:
            if i >= len(l_comp[frame]):
                i = np.random.randint(len(l_comp[frame]))
            v = l_comp[frame][i]
            # Remove from current frame
            while v in m_comp[frame]:
                _i = m_comp[frame].index(v)
                m_comp[frame].pop(_i)
                m_ref[frame].pop(_i)
            l_comp[frame].pop(i)
            # Create new trajectory
            start, end = comp_tracks[comp_tracks[:, 0] == v, 1:3][0]
            if start == end:
                comp_tracks = comp_tracks[comp_tracks[:, 0] != v]
                comp_tracks[comp_tracks[:, 3] == v, 3] = 0
            elif frame == start:
                comp_tracks[comp_tracks[:, 0] == v, 1] += 1
            elif frame == end:
                comp_tracks[comp_tracks[:, 0] == v, 2] -= 1
            else:
                comp_tracks[comp_tracks[:, 0] == v, 2] = frame - 1
                comp_tracks = np.concatenate(
                    [comp_tracks, [[next_id, frame + 1, end, v]]], axis=0)
                comp_tracks[comp_tracks[:, 0] == v, 3] = next_id
                for f in range(frame + 1, end + 1):
                    _l_comp = np.asarray(l_comp[f])
                    _m_comp = np.asarray(m_comp[f])
                    _l_comp[_l_comp == v] = next_id
                    _m_comp[_m_comp == v] = next_id
                    l_comp[f] = _l_comp.tolist()
                    m_comp[f] = _m_comp.tolist()

            next_id += 1

    # Add false positives
    if noise_add_false_positive > 0:
        label = traj["labels_comp"]
        next_id = np.max(comp_tracks[:, 0]) + 1
        max_frame = np.max(comp_tracks[:, 2])
        fp_to_add = int(noise_add_false_positive)
        np.random.seed(seed)
        for _ in range(fp_to_add):
            frame = np.random.randint(max_frame + 1)
            comp_tracks = np.concatenate(
                [comp_tracks, [[next_id, frame, frame, 0]]], axis=0)
            label[frame].append(next_id)
            next_id += 1

    # Unmatch true positives
    if noise_remove_matches > 0:
        m_comp = traj["mapped_comp"]
        m_ref = traj["mapped_ref"]
        candidates = []
        for frame in range(1, len(m_comp)):
            for i in range(len(m_comp[frame])):
                candidates.append(frame)
        np.random.seed(seed)
        np.random.shuffle(candidates)
        num_unassoc = min(noise_remove_matches, len(candidates))
        for frame in candidates[:num_unassoc]:
            total_inds = len(m_comp[frame])
            i = np.random.randint(total_inds)
            m_comp[frame].pop(i)
            m_ref[frame].pop(i)

    # Add IDSw
    if noise_add_idsw > 0:
        labels_comp = traj["labels_comp"]
        m_comp = traj["mapped_comp"]
        candidates = []
        for frame in range(len(m_comp)):
            if np.unique(m_comp[frame]).shape[0] <= 1:
                continue
            for i in range(len(np.unique(m_comp[frame])) - 1):
                candidates.append(frame)
        np.random.seed(seed)
        np.random.shuffle(candidates)
        num_unassoc = min(noise_add_idsw, len(candidates))
        for frame in candidates[:num_unassoc]:
            # Select two random indices
            comp = m_comp[frame]
            c1, c2 = np.random.choice(comp, 2, replace=False)
            end1 = int(comp_tracks[comp_tracks[:, 0] == c1, 2].squeeze())
            end2 = int(comp_tracks[comp_tracks[:, 0] == c2, 2].squeeze())
            children1 = comp_tracks[:, 3] == c1
            children2 = comp_tracks[:, 3] == c2
            # Swap the two indices
            for f in range(frame, max(end1, end2) + 1):
                _l_comp = np.asarray(labels_comp[f])
                _comp = np.asarray(m_comp[f])
                i1 = _comp == c1
                i2 = _comp == c2
                _comp[i1] = c2
                _comp[i2] = c1
                i1 = _l_comp == c1
                i2 = _l_comp == c2
                _l_comp[i1] = c2
                _l_comp[i2] = c1
                labels_comp[f] = _l_comp.tolist()
                m_comp[f] = _comp.tolist()
            i1 = comp_tracks[:, 0] == c1
            i2 = comp_tracks[:, 0] == c2
            comp_tracks[i1, 2] = end2
            comp_tracks[i2, 2] = end1
            comp_tracks[children1, 3] = c2
            comp_tracks[children2, 3] = c1

    return comp_tracks, traj


def is_new_setting(
        setting: dict,
        path: str,
        name: str,
        df=None,
):
    if exists(path):
        setting["name"] = name
        if df is None:
            df = pd.read_csv(path, index_col="index", sep=";")
        _df = df.copy()
        for k, v in setting.items():
            _df = _df[_df[k] == v]
            if len(_df) == 0:
                return True, df
        return False, df
    return True, df


def append_results(
        path: str,
        results: dict,
):
    # Check if the file exists
    results = pd.DataFrame.from_dict(results, orient="index").T
    if exists(path):
        df = pd.read_csv(path, index_col="index", sep=";")
        df = pd.concat([df, results])
        df.reset_index(drop=True, inplace=True)
    else:
        df = results
    df.to_csv(path, index_label="index", sep=";")


def evaluate_sequence(
        gt: str,
        name: str,
        threads: int = 0,
        csv_file: str = None,
        shuffle: bool = False,
):
    """
    Evaluates a single sequence

    Args:
        res: The path to the results.
        gt: The path to the ground truth.
        multiprocessing: Whether to use multiprocessing (recommended!).
        csv_file: The path to the csv file to store the results.

    Returns:
        The results stored in a dictionary.

    """

    print("Run noise test on ", gt, end="...")
    # Prepare all metrics
    metrics = copy.deepcopy(ALL_METRICS)
    metrics.remove("Valid")
    metrics.remove("SEG")

    comp_tracks, ref_tracks, traj, segm, comp_masks = load_data(gt, threads)

    # Selection of noise settings
    repeats = 10
    noise_settings = [{}]

    # # Add mitosis detection noise
    # parents, counts = np.unique(
    #     comp_tracks[comp_tracks[:, 3] > 0, 3], return_counts=True)
    # num_parents = len(parents[counts > 1])
    # for p in range(1, num_parents+1   ):
    #     for i in range(0, repeats):
    #         noise_settings.append({
    #             "seed": i,
    #             "noise_remove_mitosis": p,
    #         })
    #
    # # Add false negative noise
    num_false_negs_max = np.sum(ref_tracks[:, 2] - ref_tracks[:, 1] + 1)
    # for fn in range(1, min(500, num_false_negs_max)):
    #     for i in range(0, repeats):
    #         noise_settings.append({
    #             "seed": i,
    #             "noise_add_false_negative": fn,
    #         })
    #
    # # Add false positive noise
    # for fp in range(1, 500):
    #     for i in range(0, repeats):
    #         noise_settings.append({
    #             "seed": i,
    #             "noise_add_false_positive": fp,
    #         })


    number = 9
    csv_file = csv_file[:-4] + f"_{number}.csv"

    # Add matching noise
    for match in range(1, min(500, num_false_negs_max)):
        for i in range(0, repeats):
            if i != number:
               continue
            noise_settings.append({
                "seed": i,
                "noise_remove_matches": match,
            })

    # Add ID switch noise
    for idsw in range(1, 500):
        for i in range(0, repeats):
            if i != number:
               continue
            noise_settings.append({
                "seed": i,
                "noise_add_idsw": idsw,
            })

    # if shuffle:
    #     np.random.shuffle(noise_settings)
    df = None
    for i, setting in enumerate(noise_settings):
        print(f"\rRun noise test on ", gt, f"\t{i + 1}\t/ {len(noise_settings)}", end="")
        # Check if noise setting is new
        default_setting = {
            "seed": 0,
            "noise_add_false_positive": 0,
            "noise_add_false_negative": 0,
            "noise_add_idsw": 0,
            "noise_remove_matches": 0,
            "noise_remove_mitosis": 0,
        }

        default_setting.update(setting)

        is_new, df = is_new_setting(default_setting, csv_file, name, df)
        if not is_new:
            continue
        df = None
        # Add noise to the data and calculate the metrics
        n_comp_tracks, n_traj = add_noise(
            comp_tracks, ref_tracks, traj, **setting)

        results = dict(name=name)

        metrics = calculate_metrics(
            n_comp_tracks, ref_tracks, n_traj, segm, comp_masks, metrics, is_valid=True)
        results.update(metrics)
        results.update(default_setting)
        append_results(csv_file, results)

    print("")


def evaluate_all(
        gt_root: str,
        csv_file: str = None,
        threads: int = 0
):
    """
    Evaluate all sequences in a directory

    Args:
        gt_root: The root directory of the ground truth.
        csv_file: The path to the csv file to store the results.
        threads: The number of threads to use for multiprocessing.

    Returns:
        The results stored in a dictionary.
    """
    ret = parse_directories(gt_root, gt_root)
    i = 0
    for res, gt, name in zip(*ret):
        if 12 > i >= 0:
            evaluate_sequence(gt, name, threads, csv_file)
        i += 1


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Evaluates CTC-Sequences.')
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('-r', '--recursive', action="store_true")
    parser.add_argument('--csv-file', type=str, default=None)
    parser.add_argument('-n', '--num-threads', type=int, default=0)
    #parser.add_argument('-s', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    """
    Main function that is called when the script is executed.
    """
    args = parse_args()

    # Evaluate sequence or whole directory
    if args.recursive:
        evaluate_all(args.gt, args.csv_file, args.num_threads)
    else:
        challenge = basename(dirname(args.gt))
        sequence = basename(args.gt).replace("_GT", "")
        name = challenge + "_" + sequence
        evaluate_sequence(args.gt, name, args.num_threads, args.csv_file)


if __name__ == "__main__":
    main()
