#!/usr/bin/python3
import argparse
import os
import re
import shutil
from collections import defaultdict, deque, namedtuple

import cv2
import numpy as np
from tqdm import tqdm

from input import preprocess_input
from output import preprocess_output
from reader import RosBagReader
from utils import timestamp_to_sec

# Constants
WINDOW_DURATION = 0.2
OPPONENT_SYNC_THRESHOLD = 0.05

# Mapping from radar topic to sensor type
RADAR_SENSOR_MAPPING = {
    "/radar_front/ars548_process/detections": "front",
    "/radar_rear/ars548_process/detections": "rear",
}

RadarData = namedtuple("RadarData", ["msg", "sensor_type"])

# Regex for ground truth opponent odometry topics (e.g. "/vehicle_3/odometry")
vehicle_odom_pattern = re.compile(r"^/vehicle_(\d+)/odometry$")


def add_opponent_history(history, vehicle_id, timestamp, pose, threshold):
    """
    Add a (timestamp, pose) pair for a given vehicle into the opponent history.
    Immediately trim entries older than (timestamp - threshold).
    """
    if vehicle_id not in history:
        history[vehicle_id] = deque()
    history[vehicle_id].append((timestamp, pose))
    while history[vehicle_id] and history[vehicle_id][0][0] < timestamp - threshold:
        history[vehicle_id].popleft()
    if not history[vehicle_id]:
        del history[vehicle_id]


def trim_radar_window(radar_window, current_time, window_duration, radar_counts):
    """
    Trim radar_window so that it only contains items with timestamps >= current_time - window_duration.
    Also update the radar_counts accordingly.
    """
    while (
        radar_window
        and timestamp_to_sec(radar_window[0].msg) < current_time - window_duration
    ):
        old_radar = radar_window.popleft()
        radar_counts[old_radar.sensor_type] -= 1


def normalize(corners, grid_size=800, meter_range=200):
    # First shift coordinates from ego-relative to grid coordinates
    # Move origin from center to bottom-left by adding meter_range/2
    corners_shifted = corners + meter_range / 2

    # Convert from meters to grid coordinates (0 to grid_size)
    pixel_resolution = meter_range / grid_size
    corners_pixels = corners_shifted / pixel_resolution

    # Normalize to 0-1 range by dividing by grid size
    corners_normalized = corners_pixels / grid_size

    # Clip values to ensure they're in [0, 1]
    # corners_normalized = np.clip(corners_normalized, 0, 1)

    return corners_normalized


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process an MCAP file with radar and opponent data (ground truth or tracks)."
    )
    parser.add_argument(
        "--mcap_dir",
        type=str,
        default="/mnt/hdd/bags/uva_tum_8_4.mcap",
        help="Path to the MCAP file.",
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        default=[
            "/vehicle/uva_odometry",
            "/vehicle_3/odometry",
            "/radar_rear/ars548_process/detections",
            "/radar_front/ars548_process/detections",
        ],
        help="List of topic names to process.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/mnt/hdd/adjusted",
        help=(
            "Base directory for the dataset. The script will create "
            "subdirectories 'images' and 'labels' inside this folder."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["gt", "tracks"],
        default="gt",
        help=(
            "Mode to use for opponent data: 'gt' for ground truth odometry or "
            "'tracks' for predicted tracks."
        ),
    )
    parser.add_argument(
        "--save-timestamps",
        action="store_true",
        help="If set, saves a timestamps.txt file mapping each frame to a timestamp (in seconds).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths and topics
    MCAP_DIR = args.mcap_dir
    TOPIC_NAMES = args.topics
    DATASET = args.dataset
    mode = args.mode

    IMG_SAVE_PATH = os.path.join(DATASET, "images")
    LABEL_SAVE_PATH = os.path.join(DATASET, "labels")

    # Create output directories (will error if they already exist)
    shutil.rmtree(DATASET, ignore_errors=True)  # Delete old dataset if it exists
    os.makedirs(IMG_SAVE_PATH)
    os.makedirs(LABEL_SAVE_PATH)

    reader = RosBagReader(MCAP_DIR, TOPIC_NAMES)

    radar_window = deque()
    radar_counts = defaultdict(int)
    # A single opponent history storing tuples (timestamp, pose) per vehicle id.
    opponent_history = defaultdict(deque)

    timestamps = []
    label_num = 0

    for topic, msg, timestamp in tqdm(reader.read_messages()):
        timestamp_sec = timestamp * 1e-9  # Convert nanoseconds to seconds

        # --- Process Radar Messages ---
        if "radar" in topic:
            sensor_location = RADAR_SENSOR_MAPPING.get(topic, "unknown")
            radar_data = RadarData(msg=msg, sensor_type=sensor_location)
            radar_window.append(radar_data)
            radar_counts[sensor_location] += 1
            trim_radar_window(
                radar_window, timestamp_sec, WINDOW_DURATION, radar_counts
            )

        # --- Process Opponent Data ---
        # Ground truth opponent odometry (for mode "gt")
        if mode == "gt" and vehicle_odom_pattern.match(topic):
            match = vehicle_odom_pattern.match(topic)
            if match:
                vehicle_id = match.group(1)
                # For ground truth, extract the pose from msg.pose.pose.
                add_opponent_history(
                    opponent_history,
                    vehicle_id,
                    timestamp_sec,
                    msg.pose.pose,
                    OPPONENT_SYNC_THRESHOLD,
                )

        # Predicted tracks (for mode "tracks")
        if mode == "tracks" and "/opponent/tracks" in topic:
            # Each BatchTrack message can contain multiple tracks.
            for track in msg.tracks:
                vehicle_id = str(track.track_id)
                add_opponent_history(
                    opponent_history,
                    vehicle_id,
                    timestamp_sec,
                    track.pose.pose,
                    OPPONENT_SYNC_THRESHOLD,
                )

        # --- Process Ego Odometry ---
        if "/vehicle/uva_odometry" in topic:
            ego_time = timestamp_sec
            trim_radar_window(radar_window, ego_time, WINDOW_DURATION, radar_counts)

            # Ensure that radar data from both sensors is available.
            if not (
                radar_counts.get("front", 0) > 0 or radar_counts.get("rear", 0) > 0
            ):
                continue

            # Synchronize opponent poses to the ego timestamp.
            synced_opponent_poses = {}
            for (
                vehicle_id,
                history_deque,
            ) in opponent_history.items():
                for ts, pose in reversed(history_deque):
                    if abs(ego_time - ts) <= OPPONENT_SYNC_THRESHOLD:
                        synced_opponent_poses[vehicle_id] = pose
                        break
                    elif ts < ego_time - OPPONENT_SYNC_THRESHOLD:
                        break

            if radar_window and synced_opponent_poses:
                bounding_boxes = preprocess_output(msg.pose.pose, synced_opponent_poses)

                with open(os.path.join(LABEL_SAVE_PATH, f"{label_num}.txt"), "w") as f:
                    for bounding_box in bounding_boxes:
                        bb_distances = np.sqrt(
                            np.sum(np.square(bounding_box), axis=-1),
                        ).reshape(-1)
                        if np.any(bb_distances < 100):
                            normalized_box = normalize(bounding_box)
                            if np.all(
                                (normalized_box >= 0.0) & (normalized_box <= 1.0)
                            ):
                                bounding_box_str = f"0 {' '.join(map(str, normalized_box.reshape(-1)))}\n"
                            else:
                                continue
                        else:
                            continue
                        f.write(bounding_box_str)

                radar_grid = preprocess_input(msg, radar_window)
                radar_grid = (radar_grid * 255.0).astype(np.uint8)
                cv2.imwrite(os.path.join(IMG_SAVE_PATH, f"{label_num}.png"), radar_grid)
                timestamps.append(ego_time)
                label_num += 1
    if args.save_timestamps:
        ts_path = os.path.join(DATASET, "timestamps.txt")
        with open(ts_path, "w") as f:
            for ts in timestamps:
                f.write(f"{ts:.9f}\n")  # high precision
        print(f"Timestamps saved to {ts_path}")


if __name__ == "__main__":
    main()
