import warnings
import numpy as np


def valid_parents(
        tracks: np.ndarray,
):
    """
    Checks if all parents are >= 0.

    Args:
        tracks: The result tracks. Numpy array (n x 4) the columns are:
            0: Label
            1: Birth
            2: End
            3: Parent

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
        tracks: The result tracks. Numpy array (n x 4) the columns are:
            0: Label
            1: Birth
            2: End
            3: Parent

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
        tracks: The result tracks. Numpy array (n x 4) the columns are:
            0: Label
            1: Birth
            2: End
            3: Parent

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


def valid_ends(
        tracks: np.ndarray,
):
    """
    Checks if the end is not before the birth.

    Args:
        tracks: The result tracks. Numpy array (n x 4) with The columns are:
            0: Label
            1: Birth
            2: End
            3: Parent

    Returns:
        1 if all parent links are valid, 0 otherwise.

    """
    is_valid = 1
    for track in tracks:
        i, birth, end, parent = track
        if end < birth:
            warnings.warn(
                f"Track ends before birth: {i} {birth} {end} {parent}",
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
        masks: The mask files to inspect.
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
            warnings.warn(f"Empty frame {i}. Ok for CTMC datasets.", UserWarning)
            is_valid = 0
    return int(is_valid)


def no_empty_tracking_result(
        tracks: np.ndarray
):
    """
    Checks if there is at least one detection in th results.

    Args:
        tracks: The tracks to inspect

    Returns:
        1 if there are detections, 0 otherwise.
    """
    is_valid = 1
    if len(tracks) == 0:
        warnings.warn("No tracks in result.", UserWarning)
        is_valid = 0
    return is_valid


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
    # If tracks is empty, the result is invalid
    is_valid = no_empty_tracking_result(tracks)
    # Get the labels in each frame
    if masks is not None:
        num_frames = max(tracks[:, 2].max() + 1, len(masks))
    else:
        num_frames = tracks[:, 2].max() + 1
    frames = [[] for _ in range(num_frames)]
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
    # Check if end is not before birth
    is_valid *= valid_ends(tracks)
    # Check if all labels are in the frames they are used to be
    if masks is not None:
        is_valid *= inspect_masks(frames, masks, labels_in_frames)
    # Check if frames are empty
    no_empty_frames(frames)  # Should this make the validation irregular?
    return int(is_valid)
