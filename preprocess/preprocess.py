#!/usr/bin/python3
import argparse
import os
import re
import shutil
from collections import defaultdict, deque, namedtuple

import cv2
import numpy as np
from input import preprocess_per_vehicle
from reader import RosBagReader
from tqdm import tqdm

from utils import timestamp_to_sec

RADAR_WINDOW_DURATION = 0.2
SYNC_THRESHOLD = 0.05

RADAR_TOPIC_TO_SENSOR = {
    "/radar_front/ars548_process/detections": "front",
    "/radar_rear/ars548_process/detections": "rear",
}

RadarData = namedtuple("RadarData", ["msg", "sensor_type"])
vehicle_odom_pattern = re.compile(r"^/vehicle_(\d+)/odometry$")


def save_frame_data(frame_id, radar_grid, vehicle_data, output_dirs):
    image_dir = output_dirs["image"]
    bbox_dir = output_dirs["bbox"]
    velocity_dir = output_dirs["velocity"]

    img_path = os.path.join(image_dir, f"{frame_id}.png")
    cv2.imwrite(img_path, (radar_grid * 255).astype(np.uint8))

    velocity_dict = {}
    bbox_path = os.path.join(bbox_dir, f"{frame_id}.txt")
    with open(bbox_path, "w") as f:
        for vid, data in vehicle_data.items():
            box = data["box"]
            points = data["points"]
            velocity = data["velocity"]

            if len(points) <= 5:
                continue

            norm_box = normalize(box)
            if not np.all((0.0 <= norm_box) & (norm_box <= 1.0)):
                continue

            line = f"0 {' '.join(map(str, norm_box.reshape(-1)))}\n"
            f.write(line)

            velocity_dict[vid] = {"points": points, "velocity": velocity}

    if velocity_dict:
        np.save(
            os.path.join(velocity_dir, f"{frame_id}.npy"),
            velocity_dict,
            allow_pickle=True,
        )


def update_latest_opponent(history, vehicle_id, timestamp, pose, vel_x, vel_y):
    history[vehicle_id] = (timestamp, pose, vel_x, vel_y)


def trim_expired_radar(
    radar_window, radar_counts, current_time, window_duration=RADAR_WINDOW_DURATION
):
    while (
        radar_window
        and timestamp_to_sec(radar_window[0].msg) < current_time - window_duration
    ):
        old = radar_window.popleft()
        radar_counts[old.sensor_type] -= 1


def normalize(corners, grid_size=800, meter_range=200):
    shifted = corners + meter_range / 2
    pixels = shifted / (meter_range / grid_size)
    return pixels / grid_size


def sync_opponents_to_ego(opponent_history, ego_time):
    synced = {}
    for vehicle_id, (ts, pose, velocity_x, velocity_y) in opponent_history.items():
        if abs(ego_time - ts) <= SYNC_THRESHOLD:
            synced[vehicle_id] = (pose, velocity_x, velocity_y)
    return synced


def handle_radar_message(topic, msg, time_sec, radar_window, radar_counts):
    sensor = RADAR_TOPIC_TO_SENSOR[topic]
    radar_window.append(RadarData(msg=msg, sensor_type=sensor))
    radar_counts[sensor] += 1


def handle_opponent_odometry(topic, msg, time_sec, opponent_history):
    match = vehicle_odom_pattern.match(topic)
    if match:
        vehicle_id = match.group(1)
        update_latest_opponent(
            opponent_history,
            vehicle_id,
            time_sec,
            msg.pose.pose,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        )


def handle_ego_odometry(
    msg, time_sec, radar_window, radar_counts, opponent_history, frame_id, output_dirs
):
    trim_expired_radar(radar_window, radar_counts, time_sec)

    if not (radar_counts.get("front", 0) or radar_counts.get("rear", 0)):
        return frame_id

    synced_opponents = sync_opponents_to_ego(opponent_history, time_sec)
    if not (radar_window and synced_opponents):
        return frame_id

    ego_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
    radar_grid, vehicle_data = preprocess_per_vehicle(
        msg.pose.pose, ego_velocity, synced_opponents, radar_window
    )
    save_frame_data(frame_id, radar_grid, vehicle_data, output_dirs)
    return frame_id + 1


def process_message(topic, msg, time_sec, state, frame_id, output_dirs):
    if "radar" in topic:
        handle_radar_message(
            topic, msg, time_sec, state["radar_window"], state["radar_counts"]
        )
    elif vehicle_odom_pattern.match(topic):
        handle_opponent_odometry(topic, msg, time_sec, state["opponent_history"])
    elif "/vehicle/uva_odometry" in topic:
        frame_id = handle_ego_odometry(
            msg,
            time_sec,
            state["radar_window"],
            state["radar_counts"],
            state["opponent_history"],
            frame_id,
            output_dirs,
        )
    return frame_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training data from MCAP radar logs."
    )
    parser.add_argument(
        "--mcap_dir", type=str, default="/mnt/hdd/bags/uva_tum_8_4.mcap"
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
    )
    parser.add_argument("--dataset", type=str, default="/mnt/hdd/adjusted")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dirs = {
        "image": os.path.join(args.dataset, "images"),
        "bbox": os.path.join(args.dataset, "bounding_box_labels"),
        "velocity": os.path.join(args.dataset, "velocity"),
    }
    shutil.rmtree(args.dataset, ignore_errors=True)
    for directory in output_dirs.values():
        os.makedirs(directory)

    reader = RosBagReader(args.mcap_dir, args.topics)

    state = {
        "radar_window": deque(),
        "radar_counts": defaultdict(int),
        "opponent_history": {},
    }

    frame_id = 0
    for topic, msg, timestamp in tqdm(reader.read_messages()):
        time_sec = timestamp * 1e-9
        frame_id = process_message(topic, msg, time_sec, state, frame_id, output_dirs)


if __name__ == "__main__":
    main()
