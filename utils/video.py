import argparse
import glob
import os

import cv2
import numpy as np
from natsort import natsorted  # For natural sorting of filenames

# Constants (adjust if necessary)
GRID_SIZE = 800
METER_RANGE = 200  # Keep if needed for context AND for GT transformation


# --- UPDATED: Function to draw Ground Truth Boxes ---
def draw_ground_truth_boxes(image, label_path, grid_size, meter_range=None):
    """
    Reads normalized ground truth bounding boxes and overlays them on the image.
    Assumes coordinates are normalized (i.e., between 0 and 1).
    Boxes are drawn in green.
    """
    if not os.path.exists(label_path):
        return image  # Skip if label file doesn't exist

    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    print(f"Warning: Skipping malformed line in {label_path}: {line}")
                    continue
                try:
                    coords = list(map(float, parts[1:9]))
                    bbox = np.array(coords).reshape(4, 2)
                    bbox_pixels = (bbox * grid_size).astype(np.int32)

                    img = cv2.polylines(
                        image,
                        [bbox_pixels],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=1,
                    )
                    print(
                        "Image after polylines:",
                        img[bbox_pixels[0][1], bbox_pixels[0][0]],
                    )  # Check pixel value
                except Exception as e:
                    print(f"Error processing line in {label_path}: {line} ({e})")
                    continue
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")

    return image


def visualize_frame_for_video(
    radar_grid_img,  # Expecting image loaded by cv2 (BGR, uint8)
    relative_velocity=None,
    gt_label_path=None,
    grid_size=GRID_SIZE,
    meter_range=METER_RANGE,  # Pass meter_range for GT transformation
    scaling_factor=1,
):
    """
    Prepares a single frame for the video by drawing relative velocity text
    and optionally ground truth bounding boxes.

    Args:
        radar_grid_img (np.ndarray): A (H, W, 3) BGR uint8 image.
        relative_velocity (dict or tuple, optional): Velocity data.
        gt_label_path (str, optional): Path to the ground truth label file.
        grid_size (int): The pixel size of the grid.
        meter_range (float): The meter range covered by the grid.
        scaling_factor (int): Factor to scale the output image.

    Returns:
        np.ndarray: The image frame with overlays.
    """
    # Work on a copy
    img = radar_grid_img.copy()
    height, width, _ = img.shape
    y_pos = 20  # Starting y position for text

    # --- Draw Ground Truth Boxes (if path provided) ---
    if gt_label_path:
        # Pass meter_range to the drawing function
        img = draw_ground_truth_boxes(img, gt_label_path, grid_size, meter_range)
        # cv2.imwrite("debug_gt_test.png", img)  # Save here
        # print("Saved debug_gt_test.png")
        cv2.putText(
            img,
            "GT: Green",
            (width - 100, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y_pos += 20

    # --- Draw Velocity Text ---
    if relative_velocity is not None:
        text_lines = []
        if hasattr(relative_velocity, "items"):
            for vehicle_id, (vx, vy) in relative_velocity.items():
                text_lines.append(
                    f"ID {vehicle_id}: V_long={vx:.2f}, V_lat={vy:.2f} m/s"
                )
        elif (
            isinstance(relative_velocity, (tuple, list, np.ndarray))
            and len(relative_velocity) == 2
        ):
            vx, vy = relative_velocity
            text_lines.append(f"Relative Vel: Vx={vx:.2f}, Vy={vy:.2f} m/s")
        else:
            if (
                isinstance(relative_velocity, np.ndarray)
                and relative_velocity.size == 0
            ):
                pass
            else:
                print(
                    f"Warning: Unexpected format for relative_velocity: {type(relative_velocity)}, value: {relative_velocity}"
                )

        for i, text in enumerate(text_lines):
            cv2.putText(
                img,
                text,
                (10, y_pos + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # --- Optional Scaling ---
    if scaling_factor != 1:
        img = cv2.resize(img, (width * scaling_factor, height * scaling_factor))

    return img


def create_radar_video(
    image_dir,
    velocity_dir,
    gt_label_dir,
    output_video_path,
    plot_gt,
    fps=30.0,
    start_frame=0,
    end_frame=None,
    image_ext="png",
    grid_size=GRID_SIZE,
    meter_range=METER_RANGE,  # Pass meter_range needed for GT
):
    """
    Creates a video from radar grid images, optionally overlaying ground truth
    and relative velocity data.

    Args:
        image_dir (str): Path to the image directory.
        velocity_dir (str): Path to the velocity directory.
        gt_label_dir (str): Path to the ground truth label directory.
        output_video_path (str): Path to save the output MP4 video file.
        plot_gt (bool): If True, plot ground truth boxes.
        fps (float): Frames per second.
        start_frame (int): Start frame index.
        end_frame (int, optional): End frame index.
        image_ext (str): Image file extension.
        grid_size (int): Grid size in pixels.
        meter_range (float): Grid range in meters.
    """
    # Find image files and sort them naturally
    image_files = natsorted(glob.glob(os.path.join(image_dir, f"*.{image_ext}")))

    if not image_files:
        print(f"Error: No images found in {image_dir} with extension .{image_ext}")
        return

    # Determine frame range
    max_possible_frame = len(image_files) - 1
    if end_frame is None:
        end_frame = max_possible_frame
    else:
        end_frame = min(end_frame, max_possible_frame)

    if start_frame > end_frame:
        print(
            f"Error: start_frame ({start_frame}) is greater than end_frame ({end_frame})."
        )
        return

    image_indices_to_process = list(range(start_frame, end_frame + 1))
    num_frames = len(image_indices_to_process)

    if num_frames == 0:
        print(
            f"Error: No frames found in the specified range ({start_frame} to {end_frame})."
        )
        return

    print(
        f"Processing {num_frames} frames (from index {start_frame} to {end_frame})..."
    )

    # --- Initialize Video Writer ---
    first_image_index = image_indices_to_process[0]
    first_image_path = os.path.join(image_dir, f"{first_image_index}.{image_ext}")
    if not os.path.exists(first_image_path):
        print(f"Error: First image in range not found: {first_image_path}")
        return
    first_frame_img = cv2.imread(first_image_path)
    if first_frame_img is None:
        print(f"Error: Could not read the first image: {first_image_path}")
        return
    height, width, _ = first_frame_img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for path: {output_video_path}")
        return

    # --- Process Frames ---
    for frame_index in image_indices_to_process:
        print(f"Processing frame {frame_index}...")
        img_path = os.path.join(image_dir, f"{frame_index}.{image_ext}")

        # Load radar grid image
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found {img_path}. Skipping frame.")
            continue
        radar_img = cv2.imread(img_path)
        if radar_img is None:
            print(f"Warning: Could not read image {img_path}. Skipping frame.")
            continue

        # --- Load Velocity Data (Optional) ---
        velocity_data = None
        velocity_file_path = os.path.join(velocity_dir, f"{frame_index}.npy")
        if os.path.exists(velocity_file_path):
            try:
                loaded_vel = np.load(velocity_file_path, allow_pickle=True)
                if loaded_vel.shape == () and isinstance(loaded_vel.item(), dict):
                    velocity_data = loaded_vel.item()
                else:
                    velocity_data = loaded_vel
            except Exception as e:
                print(
                    f"Warning: Could not load or parse velocity file {velocity_file_path}: {e}"
                )
                velocity_data = None

        # --- Get GT Label Path (if plotting enabled) ---
        gt_label_path_for_frame = None
        if plot_gt:
            gt_label_path_for_frame = os.path.join(gt_label_dir, f"{frame_index}.txt")

        # --- Prepare the frame with overlays ---
        output_frame = visualize_frame_for_video(
            radar_img,
            relative_velocity=velocity_data,
            gt_label_path=gt_label_path_for_frame,
            grid_size=grid_size,
            meter_range=meter_range,  # Pass meter_range
            scaling_factor=1,
        )

        # Write frame to video
        video_writer.write(output_frame)

    # --- Cleanup ---
    video_writer.release()
    print(f"Video saved successfully to: {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from radar grid images, optionally overlaying ground truth and relative velocity."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/hdd/adjusted",
        required=True,
        help="Base path to the dataset directory containing 'images', 'velocity_labels', and optionally 'bounding_box_labels' subdirectories.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="radar_video_output.mp4",
        help="Path for the output MP4 video file.",
    )
    parser.add_argument(
        "--plot_gt",
        action="store_true",
        help="Include this flag to plot ground truth bounding boxes (from 'bounding_box_labels' directory). Assumes GT coordinates are in meters.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Index of the first frame to include in the video.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Index of the last frame to include (inclusive). If None, processes all available frames.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="png",
        help="File extension for the radar grid images (e.g., png, jpg).",
    )
    # Add arguments for constants if you want them configurable, otherwise use globals
    # parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    # parser.add_argument("--meter_range", type=float, default=METER_RANGE)

    args = parser.parse_args()

    # Use constants defined at the top
    current_grid_size = GRID_SIZE
    current_meter_range = METER_RANGE

    # --- Construct specific paths from dataset_path ---
    image_dir = os.path.join(args.dataset_path, "images")
    velocity_dir = os.path.join(args.dataset_path, "velocity_labels")
    gt_label_dir = os.path.join(args.dataset_path, "bounding_box_labels")

    # --- Validate constructed directories ---
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Base dataset directory not found: {args.dataset_path}")
        exit(1)
    if not os.path.isdir(image_dir):
        print(f"Error: 'images' subdirectory not found in {args.dataset_path}")
        exit(1)
    if not os.path.isdir(velocity_dir):
        print(f"Error: 'velocity_labels' subdirectory not found in {args.dataset_path}")
        exit(1)
    if args.plot_gt and not os.path.isdir(gt_label_dir):
        print(
            f"Error: Ground truth plotting requested (--plot_gt), but 'bounding_box_labels' subdirectory not found in {args.dataset_path}"
        )
        exit(1)

    # --- Create the video ---
    create_radar_video(
        image_dir=image_dir,
        velocity_dir=velocity_dir,
        gt_label_dir=gt_label_dir,
        output_video_path=args.output_video,
        plot_gt=args.plot_gt,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        image_ext=args.image_ext,
        grid_size=current_grid_size,  # Pass constants
        meter_range=current_meter_range,  # Pass constants
    )
