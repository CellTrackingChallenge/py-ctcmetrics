import os
from os import listdir
from os.path import join, exists, isdir
import numpy as np

from utils.num_digits import NUM_DIGITS


def parse_directories(
        input_dir,
        gt_dir,
):
    """

    """

    # Parse sequences to evaluate
    challenges = [x for x in sorted(listdir(input_dir))
                  if isdir(join(input_dir, x))]
    assert len(challenges) > 0, f"No challenges found in {input_dir}"
    sequence_appendices = ["01", "02"]
    res_dirs, gt_dirs, num_digits, names = [], [], [], []
    for challenge in challenges:
        sequences = [
            x[0:2] for x in listdir(join(input_dir, challenge)) if
            isdir(join(input_dir, challenge, x))
        ]
        for sequence in sequences:
            if sequence in sequence_appendices:
                assert challenge in NUM_DIGITS
                res_dir = join(input_dir, challenge, sequence + "_RES")
                if res_dir in res_dirs:
                    continue
                num_digits.append(NUM_DIGITS[challenge])
                res_dirs.append(res_dir)
                gt_dirs.append(join(gt_dir, challenge, sequence + "_GT"))
                names.append(challenge + "_" + sequence)

    return res_dirs, gt_dirs, num_digits, names


def read_tracking_file(path):
    """
    Reads a text file representing an acyclic graph for the whole video.
    Every line corresponds to a single track that is encoded by four numbers
    separated by a space:
        L B E P where
        L - a unique label of the track (label of markers, 16-bit positive value)
        B - a zero-based temporal index of the frame in which the track begins
        E - a zero-based temporal index of the frame in which the track ends
        P - label of the parent track (0 is used when no parent is defined)

    # Checked against test datasets -> OK

    Args:
        path: Path to the text file.

    Returns:
        A numpy array of shape (N, 4) where N is the number of tracks.
        Each row represents a track and contains four numbers: L B E P
    """
    if not exists(path):
        return None
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [x.strip().split(" ") for x in lines]
    lines = [[int(y) for y in x] for x in lines]
    return np.array(lines)


def parse_masks(dir):
    """
    Reads all frame files in a directory and returns a list of frames.

    # Checked against test datasets -> OK

    Args:
        dir: The directory to read.

    Returns:
        A sorted list of frame paths.
    """
    files = os.listdir(dir)
    files = [x for x in files if x.endswith(".tif")]
    files = [join(dir, x) for x in files]
    files = sorted(files)
    return files


