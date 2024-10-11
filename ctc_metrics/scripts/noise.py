import argparse
from os.path import join, basename, dirname, exists
import copy
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

from ctc_metrics.metrics import ALL_METRICS
from ctc_metrics.utils.filesystem import parse_directories, read_tracking_file, \
    parse_masks

from ctc_metrics.scripts.evaluate import match_computed_to_reference_masks, \
    calculate_metrics


def load_data(
        gt: str,
        threads: int = 0,
):
    """
    Load the data from the ground truth and use it as computed and reference
    data.

    Args:
        gt: The path to the ground truth.
        threads: The number of threads to use for multiprocessing.

    Returns:
        The computed and reference tracks, the trajectories and the segmentation
        masks.
    """
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


def remove_mitosis(
        comp_tracks: np.ndarray,
        num_to_remove: int,
        seed: int,
):
    """
    Remove mitosis events by removing the mother daughter relation.

    Args:
        comp_tracks: The computed tracks.
        num_to_remove: The number of mitosis events to remove.
        seed: The seed for the random number generator.

    Returns:
        The computed tracks with the mitosis events removed.
    """
    if num_to_remove == 0:
        return comp_tracks
    parents, counts = np.unique(
        comp_tracks[comp_tracks[:, 3] > 0, 3], return_counts=True)
    parents = parents[counts > 1]
    num_splits = min(num_to_remove, len(parents))
    np.random.seed(seed)
    np.random.shuffle(parents)
    for parent in parents[:num_splits]:
        comp_tracks[np.isin(comp_tracks[:, 3], parent), 3] = 0
    return comp_tracks


def sample_fn(l_comp, max_num_candidates, seed):
    """
    Sample false negatives.
    """
    candidates = []
    for frame, x in enumerate(l_comp):
        for i, _ in enumerate(x):
            candidates.append((frame, i))
    np.random.seed(seed)
    np.random.shuffle(candidates)
    num_fn = min(max_num_candidates, len(candidates))
    return candidates[:num_fn]


def add_false_negatives(
        comp_tracks: np.ndarray,
        traj: dict,
        noise_add_false_negative: int,
        seed: int
):
    """
    Add false negatives to the data.

    Args:
        comp_tracks: The computed tracks.
        traj: The trajectories.
        noise_add_false_negative: The number of false negatives to add.
        seed: The seed for the random number generator.

    Returns:
        The computed tracks and the trajectories with the false negatives added.
    """
    if noise_add_false_negative == 0:
        return comp_tracks, traj

    next_id = np.max(comp_tracks[:, 0]) + 1
    l_comp = traj["labels_comp"]
    m_comp = traj["mapped_comp"]
    m_ref = traj["mapped_ref"]
    for frame, i in sample_fn(l_comp, noise_add_false_negative, seed):
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
            comp_tracks[comp_tracks[:, 3] == v, 3] = next_id
            comp_tracks = np.concatenate(
                [comp_tracks, [[next_id, frame + 1, end, v]]], axis=0)
            for f in range(frame + 1, end + 1):
                _l_comp = np.asarray(l_comp[f])
                _m_comp = np.asarray(m_comp[f])
                _l_comp[_l_comp == v] = next_id
                _m_comp[_m_comp == v] = next_id
                l_comp[f] = _l_comp.tolist()
                m_comp[f] = _m_comp.tolist()
        next_id += 1
    return comp_tracks, traj


def add_false_positives(
        comp_tracks: np.ndarray,
        traj: dict,
        noise_add_false_positive: int,
        seed: int,
):
    """
    Add false positives to the data.

    Args:
        comp_tracks: The computed tracks.
        traj: The trajectories.
        noise_add_false_positive: The number of false positives to add.
        seed: The seed for the random number generator.

    Returns:
        The computed tracks and the trajectories with the false positives added.
    """
    if noise_add_false_positive == 0:
        return comp_tracks, traj

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

    return comp_tracks, traj


def remove_matches(
        comp_tracks: np.ndarray,
        traj: dict,
        noise_remove_matches: int,
        seed: int,
):
    """
    Remove ref-comp matches from the data.

    Args:
        comp_tracks: The computed tracks.
        traj: The trajectories.
        noise_remove_matches: The number of matches to remove.
        seed: The seed for the random number generator.

    Returns:
        The computed tracks and the trajectories with the matches removed.
    """
    if noise_remove_matches == 0:
        return comp_tracks, traj

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

    return comp_tracks, traj


def add_id_switches(
        comp_tracks: np.ndarray,
        traj: dict,
        noise_add_idsw: int,
        seed: int,
):
    """
    Add ID switches to the data.

    Args:
        comp_tracks: The computed tracks.
        traj: The trajectories.
        noise_add_idsw: The number of ID switches to add.
        seed: The seed for the random number generator.

    Returns:
        The computed tracks and the trajectories with the ID switches added.
    """
    if noise_add_idsw == 0:
        return comp_tracks, traj

    labels_comp = traj["labels_comp"]
    m_comp = traj["mapped_comp"]
    candidates = []
    for frame, x in enumerate(m_comp):
        if np.unique(x).shape[0] <= 1:
            continue
        for _ in range(len(np.unique(x)) - 1):
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


def add_noise(
        comp_tracks: np.ndarray,
        traj: dict,
        seed: int = 0,
        noise_add_false_negative: int = 0,
        noise_add_false_positive: int = 0,
        noise_add_idsw: int = 0,
        noise_remove_matches: int = 0,
        noise_remove_mitosis: int = 0,
):  #pylint: disable=too-many-arguments
    """
    Add noise to the data.

    Args:
        comp_tracks: The computed tracks.
        traj: The trajectories.
        seed: The seed for the random number generator.
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

    # Remove children of mitosis events
    comp_tracks = remove_mitosis(
        comp_tracks, noise_remove_mitosis, seed)

    # Add false negatives
    comp_tracks, traj = add_false_negatives(
        comp_tracks, traj, noise_add_false_negative, seed)

    # Add false positives
    comp_tracks, traj = add_false_positives(
        comp_tracks, traj, noise_add_false_positive, seed)

    # Unmatch true positives
    comp_tracks, traj = remove_matches(
        comp_tracks, traj, noise_remove_matches, seed)

    # Add IDSw
    comp_tracks, traj = add_id_switches(
        comp_tracks, traj, noise_add_idsw, seed)

    return comp_tracks, traj


def is_new_setting(
        setting: dict,
        path: str,
        name: str,
        df: pd.DataFrame = None,
):
    """
    Check if the setting parameter setting is already existing in the csv file.

    Args:
        setting: The setting to check.
        path: The path to the csv file.
        name: The name of the sequence.
        df: The dataframe to check.

    Returns:
        True if the setting is new, False otherwise.
    """
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
        results: list,
):
    """
    Append the results to the csv file.

    Args:
        path: The path to the csv file.
        results: The results to append.
    """
    # Check if the file exists
    results = [pd.DataFrame.from_dict(r, orient="index").T for r in results]
    if exists(path):
        df = pd.read_csv(path, index_col="index", sep=";")
        df = pd.concat([df] + results)
        df.reset_index(drop=True, inplace=True)
    else:
        df = pd.concat(results)
    df.to_csv(path, index_label="index", sep=";")


def run_noisy_sample(
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray,
        traj: dict,
        segm: dict,
        metrics: list,
        name: str,
        setting: dict,
        default_setting: dict,
):  #pylint: disable=too-many-arguments
    """
    Run a noisy sample

    Args:
        comp_tracks: The computed tracks.
        ref_tracks: The reference tracks.
        traj: The trajectories.
        segm: The segmentation masks.
        metrics: The metrics to calculate.
        name: The name of the sequence.
        setting: The noise setting.
        default_setting: The default setting without noise.

    Returns:
        The results stored in a dictionary.
    """
    # Add noise to the data and calculate the metrics
    n_comp_tracks, n_traj = add_noise(
        comp_tracks, traj, **setting)

    results = {"name": name}
    results.update(default_setting)

    resulting_metrics = calculate_metrics(
        n_comp_tracks, ref_tracks, n_traj, segm, metrics,
        is_valid=True
    )
    results.update(resulting_metrics)
    return results


def filter_existing_noise_settings(
        noise_settings: list,
        csv_file: str,
        name: str,
):
    """
    Filter and remove existing noise settings from the list of noise settings.

    Args:
        noise_settings: The list of noise settings.
        csv_file: The path to the csv file.
        name: The name of the sequence.

    Returns:
        The list of new noise settings.
    """
    df = None
    new_noise_settings = []

    for _, setting in enumerate(noise_settings):
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
        new_noise_settings.append((setting, default_setting))
    return new_noise_settings


def create_noise_settings(
        repeats: int,
        num_false_neg: int,
        num_false_pos: int,
        num_idsw: int,
        num_matches: int,
        comp_tracks: np.ndarray,
        ref_tracks: np.ndarray,
):  #pylint: disable=too-many-arguments
    """
    Create a list of noise settings that should be executed from the given
    parameters.

    Args:
        repeats: The number of repeats for each noise setting.
        num_false_neg: The number of false negatives to add.
        num_false_pos: The number of false positives to add.
        num_idsw: The number of ID switches to add.
        num_matches: The number of matches to remove.
        comp_tracks: The computed tracks.
        ref_tracks: The reference tracks.

    Returns:
        The list of noise settings.
    """
    # Extract some statistics
    parents, counts = np.unique(
        comp_tracks[comp_tracks[:, 3] > 0, 3], return_counts=True)
    num_false_negs_max = int(np.sum(ref_tracks[:, 2] - ref_tracks[:, 1] + 1))

    # Create dictionary
    noise_settings = [{}]

    for i in range(0, repeats):
        # Add mitosis detection noise
        for x in range(1, len(parents[counts > 1]) + 1):
            noise_settings.append({"seed": i, "noise_remove_mitosis": x})

        # Add false negative noise
        for x in range(1, min(num_false_neg, num_false_negs_max)):
            noise_settings.append({"seed": i, "noise_add_false_negative": x})

        # Add false positive noise
        for x in range(1, num_false_pos):
            noise_settings.append({"seed": i, "noise_add_false_positive": x})

        # Add matching noise
        for x in range(1, min(num_matches, num_false_negs_max)):
            noise_settings.append({"seed": i, "noise_remove_matches": x})

        # Add ID switch noise
        for x in range(1, num_idsw):
            noise_settings.append({"seed": i, "noise_add_idsw": x})

    return noise_settings


def evaluate_sequence(
        gt: str,
        name: str,
        threads: int = 0,
        csv_file: str = None,
        save_after: int = 20,
        repeats: int = 10,
        num_false_neg: int = 500,
        num_false_pos: int = 500,
        num_idsw: int = 500,
        num_matches: int = 500,
):  #pylint: disable=too-many-arguments
    """
    Evaluates a single sequence

    Args:
        gt: The path to the ground truth.
        name: The name of the sequence.
        threads: The number of threads to use for multiprocessing.
        csv_file: The path to the csv file to store the results.
        save_after: Save results after n runs.
        repeats: The number of repeats for each noise setting.
        num_false_neg: The number of false negatives to add.
        num_false_pos: The number of false positives to add.
        num_idsw: The number of ID switches to add.
        num_matches: The number of matches to remove.

    """

    print("Run noise test on ", gt, end="...")
    # Prepare all metrics
    metrics = copy.deepcopy(ALL_METRICS)
    metrics.remove("Valid")
    metrics.remove("SEG")

    comp_tracks, ref_tracks, traj, segm, _ = load_data(gt, threads)

    # Selection of noise settings
    noise_settings = create_noise_settings(
        repeats, num_false_neg, num_false_pos, num_idsw, num_matches,
        comp_tracks, ref_tracks
    )

    # Filter existing noise settings
    new_noise_settings = filter_existing_noise_settings(
        noise_settings, csv_file, name
    )

    # Evaluate new noise settings
    if threads == 1:
        results_list = []
        for i, (setting, default_setting) in enumerate(new_noise_settings):
            print(
                f"\rRun noise test on {gt}, \t{i + 1}\t/ {len(new_noise_settings)}",
                end=""
            )
            # Add noise to the data and calculate the metrics
            results = run_noisy_sample(
                comp_tracks, ref_tracks, traj, segm, metrics,
                name, setting, default_setting
            )
            # Aggregate results and store them every n runs
            results_list.append(results)
            if len(results_list) == save_after or i + 1 == len(new_noise_settings):
                append_results(csv_file, results_list)
                results_list = []

    else:
        threads = cpu_count() if threads == 0 else threads
        with Pool(threads) as p:
            input_list = []
            for i, (setting, default_setting) in enumerate(new_noise_settings):
                print(
                    f"\rRun noise test on {gt}, \t{i + 1}\t/ {len(new_noise_settings)}",
                    end=""
                )
                # Add noise to the data and calculate the metrics
                input_list.append((
                    comp_tracks, ref_tracks, traj, segm, metrics,
                    name, setting, default_setting
                ))
                # Process in parallel and
                if len(input_list) == save_after or i + 1 == len(new_noise_settings):
                    results_list = p.starmap(run_noisy_sample, input_list)
                    append_results(csv_file, results_list)
                    input_list = []
    print("")


def evaluate_all(
        gt_root: str,
        csv_file: str = None,
        threads: int = 0,
        **kwargs
):
    """
    Evaluate all sequences in a directory

    Args:
        gt_root: The root directory of the ground truth.
        csv_file: The path to the csv file to store the results.
        threads: The number of threads to use for multiprocessing.
        **kwargs: The noise settings.

    """
    ret = parse_directories(gt_root, gt_root)
    for _, gt, name in zip(*ret):
        evaluate_sequence(gt, name, threads, csv_file, **kwargs)


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Evaluates CTC-Sequences.')
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('-r', '--recursive', action="store_true")
    parser.add_argument('--csv-file', type=str, default=None)
    parser.add_argument('-n', '--num-threads', type=int, default=0)
    parser.add_argument('--num-false-pos', type=int, default=500)
    parser.add_argument('--num-false-neg', type=int, default=500)
    parser.add_argument('--num-idsw', type=int, default=500)
    parser.add_argument('--num-matches', type=int, default=500)
    parser.add_argument('--save-after', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    """
    Main function that is called when the script is executed.
    """
    args = parse_args()

    # Evaluate sequence or whole directory
    experiments = {
        "num_false_pos": args.num_false_pos,
        "num_false_neg": args.num_false_neg,
        "num_idsw": args.num_idsw,
        "num_matches": args.num_matches,
        "repeats": args.repeats,
        "save_after": args.save_after,
    }
    if args.recursive:
        evaluate_all(args.gt, args.csv_file, args.num_threads, **experiments)
    else:
        challenge = basename(dirname(args.gt))
        sequence = basename(args.gt).replace("_GT", "")
        name = challenge + "_" + sequence
        evaluate_sequence(
            args.gt, name, args.num_threads, args.csv_file, **experiments
        )


if __name__ == "__main__":
    main()
