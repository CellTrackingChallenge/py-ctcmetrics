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
        multiprocessing: bool = True,
):
    # Read tracking files and parse mask files
    ref_tracks = read_tracking_file(join(gt, "TRA", "man_track.txt"))
    comp_tracks = np.copy(ref_tracks)
    ref_tra_masks = parse_masks(join(gt, "TRA"))
    comp_masks = ref_tra_masks
    assert len(ref_tra_masks) > 0, f"{gt}: Ground truth masks is 0!)"

    # Match golden truth tracking masks to result masks
    traj = match_computed_to_reference_masks(
        ref_tra_masks, comp_masks, multiprocessing=multiprocessing)

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
    np.random.seed(seed)
    if noise_remove_mitosis > 0:
        assert noise_remove_mitosis <= 1
        parents, counts = np.unique(
            comp_tracks[comp_tracks[:, 3] > 0, 3], return_counts=True)
        parents = parents[counts > 1]
        num_splits = min(noise_remove_mitosis, len(parents))
        np.random.shuffle(parents)
        for parent in parents[:num_splits]:
            comp_tracks[np.isin(comp_tracks[:, 3], parent), 3] = 0

    # Add false negatives
    np.random.seed(seed)
    if noise_add_false_negative > 0:
        next_id = np.max(comp_tracks[:, 0]) + 1
        l_comp = traj["labels_comp"]
        m_comp = traj["mapped_comp"]
        m_ref = traj["mapped_ref"]
        candidates = []
        for frame in range(0, len(l_comp)):
            for i in range(len(l_comp[frame])):
                candidates.append((frame, i))
        random.shuffle(candidates)
        num_fn = min(noise_add_false_negative, len(candidates))
        for frame, i in candidates[:num_fn]:
            v = l_comp[frame][i]
            # Remove from current frame
            print(m_comp[frame], m_ref[frame], v)
            while v in m_comp[frame]:
                i = m_comp[frame].index(v)
                m_comp[frame].pop(i)
                m_ref[frame].pop(i)
            print("             ", m_comp[frame], m_ref[frame])
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
                    #print(l_comp[f])
                    l_comp[f][l_comp[f] == v] = next_id
                    m_comp[f][m_comp[f] == v] = next_id
                    #print("    ", l_comp[f])

            next_id += 1

    # Add false positives
    np.random.seed(seed)
    if noise_add_false_positive > 0:
        label = traj["labels_comp"]
        next_id = np.max(comp_tracks[:, 0]) + 1
        max_frame = np.max(comp_tracks[:, 2])
        fp_to_add = int(noise_add_false_positive)
        for _ in range(fp_to_add):
            frame = np.random.randint(max_frame + 1)
            comp_tracks = np.concatenate(
                [comp_tracks, [[next_id, frame, frame, 0]]], axis=0)
            label[frame].append(next_id)

    # Unmatch true positives
    np.random.seed(seed)
    if noise_remove_matches > 0:
        m_comp = traj["mapped_comp"]
        m_ref = traj["mapped_ref"]
        candidates = []
        for frame in range(1, len(m_comp)):
            for i in range(len(m_comp[frame])):
                candidates.append(frame)
        random.shuffle(candidates)
        num_unassoc = min(noise_remove_matches, len(candidates))
        for frame in candidates[:num_unassoc]:
            total_inds = len(m_comp[frame])
            i = np.random.randint(total_inds)
            m_comp[frame].pop(i)
            m_ref[frame].pop(i)

    # Add IDSw
    np.random.seed(seed)
    if noise_add_idsw > 0:
        labels_comp = traj["labels_comp"]
        m_comp = traj["mapped_comp"]
        candidates = []
        for frame in range(len(m_comp)):
            if np.unique(m_comp[frame]).shape[0] <= 1:
                continue
            for i in range(len(np.unique(m_comp[frame])) - 1):
                candidates.append(frame)
        random.shuffle(candidates)
        num_unassoc = min(noise_add_idsw, len(candidates))
        for frame in candidates[:num_unassoc]:
            # Select two random indices
            comp = m_comp[frame]
            c1, c2 = np.random.choice(comp, 2, replace=False)
            end1 = comp_tracks[comp_tracks[:, 0] == c1, 2]
            end2 = comp_tracks[comp_tracks[:, 0] == c2, 2]
            children1 = comp_tracks[:, 3] == c1
            children2 = comp_tracks[:, 3] == c2
            # Swap the two indices
            for f in range(frame, max(end1, end2) + 1):
                _l_comp = labels_comp[f]
                _comp = m_comp[f]
                i1 = _comp == c1
                i2 = _comp == c2
                _comp[i1] = c2
                _comp[i2] = c1
                i1 = _l_comp == c1
                i2 = _l_comp == c2
                _l_comp[i1] = c2
                _l_comp[i2] = c1
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
):
    if exists(path):
        setting["name"] = name
        df = pd.read_csv(path, index_col="index", sep=";")
        for k, v in setting.items():
            df = df[df[k] == v]
            if len(df) == 0:
                return True
        return False
    return True


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
        multiprocessing: bool = True,
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

    # Selection of noise settings
    repeats = 1
    noise_settings = [{}]
    # Add ... noise
    for p in range(1, 101):
        for i in range(repeats):
            # noise_settings.append({
            #     "seed": i,
            #     "noise_mitosis_remove_one_child": p / 100,
            # })
            # noise_settings.append({
            #     "seed": i,
            #     "noise_mitosis_remove_all_children": p / 100,
            # })
            # noise_settings.append({
            #     "seed": i,
            #     "noise_add_false_positive": p / 100,
            # })
            # noise_settings.append({
            #     "seed": i,
            #     "noise_unassociate_true_positive": p / 100,
            # })
            pass


    noise_settings.append({
        "seed": 0,
        "noise_remove_mitosis": 1,
    })
    noise_settings.append({
        "seed": 0,
        "noise_add_false_positive": 1,
    })
    noise_settings.append({
        "seed": 0,
        "noise_add_false_negative": 1,
    })
    noise_settings.append({
        "seed": 0,
        "noise_remove_matches": 1,
    })
    noise_settings.append({
        "seed": 0,
        "noise_add_idsw": 1,
    })


    if shuffle:
        np.random.shuffle(noise_settings)

    # Prepare all metrics
    metrics = copy.deepcopy(ALL_METRICS)
    metrics.remove("Valid")
    metrics.remove("SEG")

    comp_tracks, ref_tracks, traj, segm, comp_masks = load_data(
        gt, multiprocessing)

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

        if not is_new_setting(default_setting, csv_file, name):
            continue

        # Add noise to the data and calculate the metrics
        n_comp_tracks, n_traj = add_noise(
            comp_tracks, ref_tracks, traj, **setting)

        results = calculate_metrics(
            n_comp_tracks, ref_tracks, n_traj, segm, comp_masks, metrics)

        results["name"] = name
        results.update(default_setting)
        append_results(csv_file, results)

    print("")


def evaluate_all(
        gt_root: str,
        multiprocessing: bool = True,
        csv_file: str = None,
):
    """
    Evaluate all sequences in a directory

    Args:
        gt_root: The root directory of the ground truth.
        multiprocessing: Whether to use multiprocessing (recommended!).
        csv_file: The path to the csv file to store the results.

    Returns:
        The results stored in a dictionary.
    """
    ret = parse_directories(gt_root, gt_root)
    for res, gt, name in zip(*ret):
        evaluate_sequence(gt, name, multiprocessing, csv_file)


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Evaluates CTC-Sequences.')
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('-r', '--recursive', action="store_true")
    parser.add_argument('--csv-file', type=str, default=None)
    parser.add_argument('--single-process', action="store_true")
    args = parser.parse_args()
    return args


def main():
    """
    Main function that is called when the script is executed.
    """
    args = parse_args()

    # Evaluate sequence or whole directory
    multiprocessing = not args.single_process
    if args.recursive:
        evaluate_all(args.gt, multiprocessing, args.csv_file)
    else:
        challenge = basename(dirname(args.gt))
        sequence = basename(args.gt).replace("_GT", "")
        name = challenge + "_" + sequence
        evaluate_sequence(args.gt, name, multiprocessing, args.csv_file)


if __name__ == "__main__":
    main()
