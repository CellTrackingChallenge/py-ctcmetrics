import warnings
import numpy as np


def valid_parents(
        tracks: np.ndarray,
):
    """
    Checks if all parents are >= 0.

    Args:
        tracks: The result tracks.

    Returns:
        1 if all parents are >= 0, 0 otherwise.

    """
    is_valid = 1
    for track in tracks:
        if track[3] < 0:
            warnings.warn(f"Invalid parent: {track}", UserWarning)
            is_valid = 0
    return int(is_valid)


def unique_labels(
        tracks: np.ndarray,
):
    """
    Checks if all labels are unique.

    Args:
        tracks: The result tracks.

    Returns:
        1 if all labels are unique, 0 otherwise.

    """
    is_valid = 1
    labels = tracks[:, 0]
    if len(np.unique(labels)) != len(labels):
        warnings.warn("Labels are not unique.", UserWarning)
        is_valid = 0
    return int(is_valid)


def valid_parent_links(
        tracks: np.ndarray,
):
    """
    Checks if all parent links are valid.

    Args:
        tracks: The result tracks.

    Returns:
        1 if all parent links are valid, 0 otherwise.

    """
    is_valid = 1
    for track in tracks:
        _, birth, _, parent = track
        if parent != 0:
            parent_idx = np.argwhere(tracks[:, 0] == parent).squeeze()
            assert parent_idx.size == 1, parent_idx
            parent_track = tracks[parent_idx]
            _, _, parent_end, _ = parent_track
            if parent_end >= birth:
                warnings.warn(
                    f"Parent ends after child starts: {track} {parent_track}",
                    UserWarning)
                is_valid = 0
    return int(is_valid)


def inspect_masks(
        frames: list,
        masks: list,
        labels_in_frames: list,
):
    """
    Inspects the masks for invalid labels.

    Args:
        frames: The frames to inspect.
        masks: The masks to inspect.
        labels_in_frames: The present labels corresponding to the file in
            "masks".

    Returns:
        1 if all labels are valid, 0 otherwise.

    """
    is_valid = 1
    for labels_in_frame, file, frame in zip(labels_in_frames, masks, frames):
        for label in labels_in_frame:
            if label != 0:
                if label not in frame:
                    warnings.warn(
                        f"Label {label} in mask but not in res_track {file}.",
                        UserWarning)
                    is_valid = 0
        for label in frame:
            if label not in labels_in_frame:
                warnings.warn(
                    f"Label {label} in res_track but not in mask {file}.",
                    UserWarning)
                is_valid = 0
    return int(is_valid)


def no_empty_frames(
        frames: list,
):
    """
    Checks if there are empty frames.

    Args:
        frames: The frames to inspect.

    Returns:
        1 if there are no empty frames, 0 otherwise.

    """
    is_valid = 1
    for i, f in enumerate(frames):
        if len(f) == 0:
            warnings.warn(f"Empty frame {i}.", UserWarning)
            is_valid = 0
    return int(is_valid)


def valid(
        masks: list,
        tracks: np.ndarray,
        labels_in_frames: list,
):
    """
    Checks if the cell tracking result is valid. The result is valid if...
    - ...all parents are >= 0
    - ...all labels are unique
    - ...parents endet before children are born
    (- ...all labels are used)
    - ...all labels are in the frames they are used to be
    - ...frames are not empty

    Args:
        masks: The masks of the result.
        tracks: The result tracks.
        labels_in_frames: The present labels corresponding to the file in
            "masks".

    Returns:
        1 if the result is valid, 0 otherwise.

    """
    is_valid = 1
    # Get the labels in each frame
    frames = [[] for _ in range(len(masks))]
    for track in tracks:
        label, birth, end, _ = track
        for frame in range(birth, end + 1):
            frames[frame].append(label)
    # Check if all parents are >= 0
    is_valid *= valid_parents(tracks)
    # Check if all labels are unique
    is_valid *= unique_labels(tracks)
    # Check if parents ends before children are born
    is_valid *= valid_parent_links(tracks)
    # Check if all labels are in the frames they are used to be
    is_valid *= inspect_masks(frames, masks, labels_in_frames)
    # Check if frames are empty
    is_valid *= no_empty_frames(frames)
    return int(is_valid)
