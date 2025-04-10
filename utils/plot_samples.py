#!/usr/bin/env python3
"""
This script processes radar images and overlays both ground truth and predicted oriented bounding boxes on them.
- Ground truth boxes are read from associated label files (each file should contain one line per vehicle in the format:
    class x1 y1 x2 y2 x3 y3 x4 y4
  with the eight numbers representing the four (x,y) pairs). These are drawn in green.
- If a YOLO model weights file is provided via --model, the model is run on each image and the predicted boxes are
  extracted from result.obb.xyxyxyxy (expected to be an array/tensor of shape (N, 4, 2), or (0, 4, 2) if no boxes)
  and drawn in red.

A legend is added at the top of each image:
  - If a model is provided: "Predicted: Red | Ground Truth: Green"
  - Otherwise: "Ground Truth: Green"

Both single-image and video processing are supported.
"""

import argparse
import os

import cv2
import numpy as np

# For model-based prediction (if requested)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = (
        None  # Will error out later if --model is used and ultralytics is not installed
    )

# Global constants
GRID_SIZE = 800  # Size of the radar grid in pixels
METER_RANGE = 200  # Range in meters
PIXEL_RESOLUTION = METER_RANGE / GRID_SIZE  # Meters per pixel


def add_legend(image, model_used):
    """
    Adds a legend to the top of the image.
    If model_used is True, the legend reads "Predicted: Red | Ground Truth: Green".
    Otherwise, it reads "Ground Truth: Green".
    """
    if model_used:
        text = "Predicted: Red | Ground Truth: Green"
    else:
        text = "Ground Truth: Green"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = 5
    # Define rectangle coordinates for a filled background
    top_left = (5, 5)
    bottom_right = (5 + text_size[0] + 2 * margin, 5 + text_size[1] + 2 * margin)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), thickness=-1)
    text_org = (top_left[0] + margin, bottom_right[1] - margin)
    cv2.putText(
        image, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
    return image


def draw_ground_truth_boxes(image, label_path):
    """
    Reads all ground truth bounding boxes from the label file and overlays them on the image.
    Each line in the label file is expected to have 9 entries:
      class x1 y1 x2 y2 x3 y3 x4 y4
    The coordinates are assumed to be normalized (0 to 1) and are scaled by GRID_SIZE.
    Boxes are drawn in green.
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return image

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # Skip if not enough data
            try:
                # Skip the first element (class) and parse the rest as floats.
                coords = list(map(float, parts[1:9]))
                bbox = np.array(coords).reshape(4, 2)
            except Exception as e:
                print(f"Error processing line in {label_path}: {line} ({e})")
                continue

            pts = (bbox * GRID_SIZE).astype(np.int32)
            image = cv2.polylines(
                image, [pts], isClosed=True, color=(0, 255, 0), thickness=1
            )
    return image


def draw_predicted_boxes(image, pred_boxes):
    """
    Draws predicted oriented bounding boxes (OBBs) on the image.
    The pred_boxes are expected to be either:
      - a numpy array (or tensor) of shape (N, 4, 2) (N boxes, each defined by 4 (x,y) pairs), or
      - a single box of shape (4,2) (which is expanded to (1, 4, 2)).
    Each box is scaled by GRID_SIZE.
    Boxes are drawn in red.
    If no boxes are predicted (i.e. shape is (0, 4, 2)), a message is printed.
    """
    if pred_boxes is None:
        return image

    # Convert tensor to numpy if necessary.
    try:
        if hasattr(pred_boxes, "cpu"):
            pred_boxes = pred_boxes.cpu().numpy()
        else:
            pred_boxes = np.array(pred_boxes)
    except Exception as e:
        print("Error converting predictions to numpy array:", e)
        return image

    # If a single box is provided, ensure it is in the shape (1, 4, 2)
    if pred_boxes.ndim == 2 and pred_boxes.shape == (4, 2):
        pred_boxes = np.expand_dims(pred_boxes, axis=0)
    elif pred_boxes.ndim != 3 or pred_boxes.shape[1:] != (4, 2):
        print("Unexpected shape for predicted OBBs:", pred_boxes.shape)
        return image

    if pred_boxes.shape[0] == 0:
        print("No predicted bounding boxes for this image.")
        return image

    for box in pred_boxes:
        pts = box.astype(np.int32)
        image = cv2.polylines(
            image, [pts], isClosed=True, color=(0, 0, 255), thickness=1
        )
    return image


def process_single_image(dataset_path, idx, ext, model=None):
    """
    Process a single image (with index idx):
      - Always overlay ground truth boxes from the label file.
      - If a model is provided, also run predictions and overlay predicted boxes.
      - Add a legend at the top of the image.
    """
    image_path = os.path.join(dataset_path, "images", f"{idx}.{ext}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file: {image_path}")
        return

    # Overlay predicted boxes (if model provided)
    if model is not None:
        results = model(image_path)
        # There can be multiple results per image.
        for result in results:
            try:
                preds = result.obb.xyxyxyxy  # Expected shape: (N, 4, 2)
            except Exception as e:
                print("Error extracting predicted boxes:", e)
                preds = None
            image = draw_predicted_boxes(image, preds)

    # Overlay ground truth boxes
    label_path = os.path.join(dataset_path, "labels", f"{idx}.txt")
    image = draw_ground_truth_boxes(image, label_path)

    # Add legend on top
    image = add_legend(image, model_used=(model is not None))

    output_image_path = os.path.join(dataset_path, f"processed_{idx}.{ext}")
    cv2.imwrite(output_image_path, image)
    print(f"Processed image with index {idx}. Saved output to: {output_image_path}")

    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(dataset_path, start_idx, end_idx, ext, output_video, fps, model=None):
    """
    Process a range of images (from start_idx to end_idx) and write the sequence to a video.
    For each image:
      - Overlay ground truth boxes from the label file.
      - If a model is provided, run predictions and overlay predicted boxes.
      - Add a legend on top.
    """
    indices = list(range(start_idx, end_idx + 1))
    if not indices:
        print("No indices to process.")
        return

    first_image_path = os.path.join(dataset_path, "images", f"{indices[0]}.{ext}")
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read image file: {first_image_path}")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(
        f"Creating video from index {start_idx} to {end_idx} and saving to: {output_video}"
    )

    for idx in indices:
        image_path = os.path.join(dataset_path, "images", f"{idx}.{ext}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image file: {image_path}. Skipping.")
            continue

        if model is not None:
            results = model(image_path)
            for result in results:
                try:
                    preds = result.obb.xyxyxyxy  # Expected shape: (N, 4, 2)
                except Exception as e:
                    print("Error extracting predicted boxes:", e)
                    preds = None
                image = draw_predicted_boxes(image, preds)

        label_path = os.path.join(dataset_path, "labels", f"{idx}.txt")
        image = draw_ground_truth_boxes(image, label_path)

        # Add legend on top
        image = add_legend(image, model_used=(model is not None))

        out.write(image)
        cv2.imshow("Video Frame", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Video processing interrupted by user.")
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"Finished video processing. Output video saved to: {output_video}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay both predicted (red) and ground truth (green) bounding boxes on radar images and save as an image or video."
    )

    parser.add_argument(
        "-p",
        "--dataset-path",
        default="/mnt/hdd/adjusted",
        help="Path to directory containing 'images/' and 'labels/' subdirectories.",
    )
    parser.add_argument(
        "-ext",
        "--extension",
        default="png",
        help="Image file extension (default: png).",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=float,
        default=100.0,
        help="Frames per second for the output video (default: 20.0).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.mp4",
        help="Output video file (used only when processing multiple images).",
    )
    parser.add_argument(
        "-s", "--start-idx", type=int, default=0, help="Start index (integer)."
    )
    parser.add_argument(
        "-e",
        "--end-idx",
        type=int,
        default=0,
        help="End index (integer). If equal to start index, a single image is processed.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="(Optional) Path to a YOLO model weights file for prediction mode. "
        "If provided, predicted boxes (red) will be overlaid along with ground truth (green).",
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path
    ext = args.extension
    output_video = args.output
    start_idx = args.start_idx
    end_idx = args.end_idx
    fps = args.fps

    if not os.path.isdir(dataset_path):
        print(f"Error: Provided dataset path does not exist: {dataset_path}")
        return

    model = None
    if args.model is not None:
        if YOLO is None:
            print(
                "Error: ultralytics is not installed. Install it to use model predictions."
            )
            return
        if not os.path.exists(args.model):
            print(f"Error: Provided model file does not exist: {args.model}")
            return
        print(f"Loading model from: {args.model}")
        model = YOLO(args.model)

    if start_idx == end_idx:
        process_single_image(dataset_path, start_idx, ext, model)
    else:
        process_video(dataset_path, start_idx, end_idx, ext, output_video, fps, model)


if __name__ == "__main__":
    main()
