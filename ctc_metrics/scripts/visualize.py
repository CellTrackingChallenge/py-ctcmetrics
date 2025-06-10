import argparse
from os import listdir, makedirs
from os.path import join
import tifffile as tiff
import cv2
import numpy as np

from ctc_metrics.utils.filesystem import read_tracking_file


SHOW_BORDER = True
BORDER_WIDTH = {
    "BF-C2DL-HSC": 25,
    "BF-C2DL-MuSC": 25,
    "Fluo-N2DL-HeLa": 25,
    "PhC-C2DL-PSC": 25,
    "Fluo-N2DH-SIM+": 0,
    "DIC-C2DH-HeLa": 50,
    "Fluo-C2DL-Huh7": 50,
    "Fluo-C2DL-MSC": 50,
    "Fluo-N2DH-GOWT1": 50,
    "PhC-C2DH-U373": 50,
}

np.random.seed(0)
PALETTE = np.random.randint(0, 256, (10000, 3))


def get_palette_color(i):
    i = i % PALETTE.shape[0]
    return PALETTE[i]


def visualize(
        img_dir: str,
        res_dir: str,
        viz_dir: str = None,
        video_name: str = None,
        border_width: str = None,
        show_labels: bool = True,
        show_parents: bool = True,
        ids_to_show: list = None,
        start_frame: int = 0,
        framerate: int = 30,
        opacity: float = 0.5,
):  # pylint: disable=too-many-arguments,too-many-locals
    """
    Visualizes the tracking results.

    Args:
        img_dir: str
            The path to the images.
        res_dir: str
            The path to the results.
        viz_dir: str
            The path to save the visualizations.
        video_name: str
            The path to the video if a video should be created. Note that no
            visualization is available during video creation.
        border_width: str or int
            The width of the border. Either an integer or a string that
            describes the challenge name.
        show_labels: bool
            Print instance labels to the output.
        show_parents: bool
            Print parent labels to the output.
        ids_to_show: list
            The IDs of the instances to show. All others will be ignored.
        start_frame: int
            The frame to start the visualization.
        framerate: int
            The framerate of the video.
        opacity: float
            The opacity of the instance colors.

    """
    # Define initial video parameters
    wait_time = max(1, round(1000 / framerate))
    if border_width is None:
        border_width = 0
    elif isinstance(border_width, str):
        try:
            border_width = int(border_width)
        except ValueError as exc:
            if border_width in BORDER_WIDTH:
                border_width = BORDER_WIDTH[border_width]
            else:
                raise ValueError(
                    f"Border width '{border_width}' not recognized. "
                    f"Existing datasets: {BORDER_WIDTH.keys()}"
                ) from exc

    # Load image and tracking data
    images = [x for x in sorted(listdir(img_dir)) if x.endswith(".tif")]
    results = [x for x in sorted(listdir(res_dir)) if x.endswith(".tif")]
    parents = {
        l[0]: l[3] for l in read_tracking_file(join(res_dir, "res_track.txt"))
    }

    # Create visualization directory
    if viz_dir:
        makedirs(viz_dir, exist_ok=True)

    video_writer = None

    # Loop through all images
    while start_frame < len(images):
        # Read image file
        img_name, res_name = images[start_frame], results[start_frame]
        img_path, res_path,  = join(img_dir, img_name), join(res_dir, res_name)
        print(f"\rFrame {img_name} (of {len(images)})", end="")

        # Visualize the image
        img = tiff.imread(img_path)
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip((img - p1) / max(p99 - p1, 1e-5) * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        viz = create_colored_image(
            img,
            tiff.imread(res_path),
            labels=show_labels,
            frame=start_frame,
            parents=parents if show_parents else None,
            ids_to_show=ids_to_show,
            opacity=opacity,
        )
        if border_width > 0:
            viz = cv2.rectangle(
                viz,
                (border_width, border_width),
                (viz.shape[1] - border_width, viz.shape[0] - border_width),
                (0, 0, 255), 1
            )

        # Save the visualization
        if video_name is not None:
            if video_writer is None:
                video_path = join(
                    viz_dir, f"{video_name.replace('.mp4', '')}.mp4")
                video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    framerate,
                    (viz.shape[1], viz.shape[0])
                )
            video_writer.write(viz)
            start_frame += 1
            continue

        # Show the video
        cv2.imshow("VIZ", viz)
        key = cv2.waitKey(wait_time)
        if key == ord("q"):
            # Quit the visualization
            break
        if key == ord("w"):
            # Start or stop the auto visualization
            if wait_time == 0:
                wait_time = max(1, round(1000 / framerate))
            else:
                wait_time = 0
        elif key == ord("d"):
            # Move to the next frame
            start_frame += 1
            wait_time = 0
        elif key == ord("a"):
            # Move to the previous frame
            start_frame -= 1
            wait_time = 0
        elif key == ord("l"):
            # Toggle the show labels option
            show_labels = not show_labels
        elif key == ord("p"):
            # Toggle the show parents option
            show_parents = not show_parents
        elif key == ord("s"):
            # Save the visualization
            if viz_dir is None:
                print("Please define the '--viz' argument to save the "
                      "visualizations.")
                continue
            viz_path = join(viz_dir, img_name) + ".jpg"
            cv2.imwrite(viz_path, viz)
        else:
            # Move to the next frame
            start_frame += 1


def create_colored_image(
        img: np.ndarray,
        res: np.ndarray,
        labels: bool = False,
        opacity: float = 0.5,
        ids_to_show: list = None,
        frame: int = None,
        parents: dict = None,
):
    """
    Creates a colored image from the input image and the results.

    Args:
        img: np.ndarray
            The input image.
        res: np.ndarray
            The results.
        labels: bool
            Print instance labels to the output.
        opacity: float
            The opacity of the instance colors.
        ids_to_show: list
            The IDs of the instances to show. All others will be ignored.
        frame: int
            The frame number.
        parents: dict
            The parent dictionary.

    Returns:
        The colored image.
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for i in np.unique(res):
        if i == 0:
            continue
        if ids_to_show is not None:
            if i not in ids_to_show:
                continue
        mask = res == i
        contour = (mask * 255).astype(np.uint8) - \
                  cv2.erode((mask * 255).astype(np.uint8), kernel)
        contour = contour != 0
        img[mask] = (
            np.round((1 - opacity) * img[mask] + opacity * get_palette_color(i))
        )
        img[contour] = get_palette_color(i)
        if frame is not None:
            cv2.putText(img, str(frame), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if labels:
            # Print label to the center of the object
            y, x = np.where(mask)
            y, x = np.mean(y), np.mean(x)
            text = str(i)
            if parents is not None:
                if i in parents:
                    if parents[i] != 0:
                        text += f"({parents[i]})"
            cv2.putText(img, text, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def parse_args():
    """ Parses the arguments. """
    parser = argparse.ArgumentParser(description='Validates CTC-Sequences.')
    parser.add_argument(
        '--img', type=str, required=True,
        help='The path to the images.'
    )
    parser.add_argument(
        '--res', type=str, required=True, help='The path to the results.'
    )
    parser.add_argument(
        '--viz', type=str, default=None,
        help='The path to save the visualizations.'
    )
    parser.add_argument(
        '--video-name', type=str, default=None,
        help='The path to the video if a video should be created. Note that no '
             'visualization is available during video creation.'
    )
    parser.add_argument(
        '--border-width', type=str, default=None,
        help='The width of the border. Either an integer or a string that '
             'describes the challenge name.'
    )
    parser.add_argument(
        '--show-no-labels', action="store_false",
        help='Print no instance labels to the output.'
    )
    parser.add_argument(
        '--show-no-parents', action="store_false",
        help='Print no parent labels to the output.'
    )
    parser.add_argument(
        '--ids-to-show', type=int, nargs='+', default=None,
        help='The IDs of the instances to show. All others will be ignored.'
    )
    parser.add_argument(
        '--start-frame', type=int, default=0,
        help='The frame to start the visualization.'
    )
    parser.add_argument(
        '--framerate', type=int, default=10,
        help='The framerate of the video.'
    )
    parser.add_argument(
        '--opacity', type=float, default=0.5,
        help='The opacity of the instance colors.'
    )
    args = parser.parse_args()
    return args


def main():
    """
    Main function that is called when the script is executed.
    """
    args = parse_args()
    visualize(
        args.img,
        args.res,
        viz_dir=args.viz,
        video_name=args.video_name,
        border_width=args.border_width,
        show_labels=args.show_no_labels,
        show_parents=args.show_no_parents,
        ids_to_show=args.ids_to_show,
        start_frame=args.start_frame,
        framerate=args.framerate,
        opacity=args.opacity,
    )


if __name__ == "__main__":
    main()
