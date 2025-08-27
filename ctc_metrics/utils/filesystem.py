import os
from os import listdir
from os.path import join, exists, isdir
import numpy as np


def load_ctmc_bounding_boxes(
        input_file: str
):
    # Load BBoxes
    assert exists(input_file), f"{input_file} does not exist!"
    with open(input_file, "r") as f:
        lines = f.readlines()
    bboxes = [[int(y) for y in x.split(",")[0:6]] for x in lines]  # Frame, ID, x, y, w, h,
    # Sort by frames
    max_frame = max([x[0] for x in bboxes])
    _frames = [[] for _ in range(max_frame+1)]
    for b in bboxes:
        _frames[b[0]].append(b[1:])
    # Sort every frame entry by its id
    frames = []
    for frame in _frames:
        inds = np.argsort([x[0] for x in frame])
        _frame = []
        for ind in inds:
            _frame.append(frame[ind])
        frames.append(_frame)

    return frames


def parse_directories(
        input_dir: str,
        gt_dir: str = None,
):
    """
    Parses a directory and searches for challenges and their respective
    result/ground-truth subdirectories.

    Args:
        input_dir: The directory to parse.
        gt_dir: The directory containing the ground truth.

    Returns:
        A tuple of three lists containing the result directories, the
        ground-truth directories and the names of the challenges/sequences.
    """

    # Parse sequences to evaluate
    challenges = [x for x in sorted(listdir(input_dir))
                  if isdir(join(input_dir, x))]
    assert len(challenges) > 0, f"No challenges found in {input_dir}"
    sequence_appendices = ["01", "02"]
    res_dirs, gt_dirs, names = [], [], []
    for challenge in challenges:
        sequences = [
            x[0:2] for x in sorted(listdir(join(input_dir, challenge))) if
            isdir(join(input_dir, challenge, x))
        ]
        for sequence in sequences:
            if sequence in sequence_appendices:
                res_dir = join(input_dir, challenge, sequence + "_RES")
                if res_dir in res_dirs:
                    continue
                res_dirs.append(res_dir)
                if gt_dir is not None:
                    _gt_dir = join(gt_dir, challenge, sequence + "_GT")
                    if not exists(_gt_dir):
                        _gt_dir = None
                    gt_dirs.append(_gt_dir)
                else:
                    gt_dirs.append(None)
                names.append(challenge + "_" + sequence)
    return res_dirs, gt_dirs, names


def read_tracking_file(
        path: str,
):
    """
    Reads a text file representing an acyclic graph for the whole video.
    Every line corresponds to a single track that is encoded by four numbers
    separated by a space:
        L B E P where
        L - a unique label of the track (label of markers, 16-bit positive value)
        B - a zero-based temporal index of the frame in which the track begins
        E - a zero-based temporal index of the frame in which the track ends
        P - label of the parent track (0 is used when no parent is defined)

    Args:
        path: Path to the text file.

    Returns:
        A numpy array of shape (N, 4) where N is the number of tracks.
        Each row represents a track and contains four numbers: L B E P
    """
    if not exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) == 0:
        return np.zeros((0, 4))
    seperator = " " if " " in lines[0] else "\t"
    lines = [x.strip().split(seperator) for x in lines]
    lines = [[int(y) for y in x if y != ""] for x in lines]
    return np.array(lines)


def parse_masks(
        directory: str
):
    """
    Reads all frame files in a directory and returns a list of frames.

    Args:
        directory: The directory to read.

    Returns:
        A sorted list of frame paths.
    """
    files = sorted(os.listdir(directory))
    files = [x for x in files if x.endswith(".tif")]
    _files = []
    for x in files:
        if x.count("_") == 3:
            # This is a 3D mask file with slices. Remove the slice number.
            x = "_".join(x.split("_")[0:3]) + ".tif"
        if x not in _files:
            _files.append(join(directory, x))
    files = sorted(_files)
    return files
