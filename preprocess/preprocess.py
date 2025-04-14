#!/usr/bin/python3
import argparse
import os
import re
import shutil
from collections import defaultdict, deque, namedtuple

import cv2
import numpy as np
from input import preprocess_input
from output import preprocess_bounding_boxes, preprocess_velocities
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


def save_frame_data(
    frame_id, bounding_boxes, radar_grid, points, velocities, output_dirs
):
    image_dir, bbox_dir, velocity_dir = (
        output_dirs["image"],
        output_dirs["bbox"],
        output_dirs["velocity"],
    )

    radar_grid_uint8 = (radar_grid * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(image_dir, f"{frame_id}.png"), radar_grid_uint8)

    velocity_save_data = {}

    with open(os.path.join(bbox_dir, f"{frame_id}.txt"), "w") as bbox_file:
        for vehicle_id, box in bounding_boxes:
            distances = np.linalg.norm(box, axis=-1).reshape(-1)
            if not np.any(distances < 100):
                continue

            norm_box = normalize(box)
            if not np.all((0.0 <= norm_box) & (norm_box <= 1.0)):
                continue

            line = f"0 {' '.join(map(str, norm_box.reshape(-1)))}\n"
            bbox_file.write(line)

            if len(points[vehicle_id]) > 5:
                velocity_save_data[vehicle_id] = {
                    "points": points[vehicle_id],
                    "velocity": velocities[vehicle_id],
                }

    if velocity_save_data:
        np.save(
            os.path.join(velocity_dir, f"{frame_id}.npy"),
            velocity_save_data,
            allow_pickle=True,
        )


def add_opponent_to_history(
    history, vehicle_id, timestamp, pose, threshold, vel_x, vel_y
):
    if vehicle_id not in history:
        history[vehicle_id] = deque()
    history[vehicle_id].append((timestamp, pose, vel_x, vel_y))
    while history[vehicle_id] and history[vehicle_id][0][0] < timestamp - threshold:
        history[vehicle_id].popleft()
    if not history[vehicle_id]:
        del history[vehicle_id]


def trim_radar_window(radar_window, current_time, window_duration, radar_counts):
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
    for vehicle_id, history in opponent_history.items():
        for ts, pose, velocity_x, velocity_y in reversed(history):
            if abs(ego_time - ts) <= SYNC_THRESHOLD:
                synced[vehicle_id] = (pose, velocity_x, velocity_y)
                break
            elif ts < ego_time - SYNC_THRESHOLD:
                break
    return synced


def process_message(
    topic,
    msg,
    time_sec,
    radar_window,
    radar_counts,
    opponent_history,
    frame_id,
    output_dirs,
):
    if "radar" in topic:
        sensor = RADAR_TOPIC_TO_SENSOR.get(topic, "unknown")
        radar_window.append(RadarData(msg=msg, sensor_type=sensor))
        radar_counts[sensor] += 1
        trim_radar_window(radar_window, time_sec, RADAR_WINDOW_DURATION, radar_counts)

    elif vehicle_odom_pattern.match(topic):
        vehicle_id = vehicle_odom_pattern.match(topic).group(1)
        add_opponent_to_history(
            opponent_history,
            vehicle_id,
            time_sec,
            msg.pose.pose,
            SYNC_THRESHOLD,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        )

    elif "/vehicle/uva_odometry" in topic:
        ego_time = time_sec
        trim_radar_window(radar_window, ego_time, RADAR_WINDOW_DURATION, radar_counts)

        if not (radar_counts.get("front", 0) or radar_counts.get("rear", 0)):
            return frame_id

        synced_opponents = sync_opponents_to_ego(opponent_history, ego_time)
        if radar_window and synced_opponents:
            bounding_boxes = preprocess_bounding_boxes(msg.pose.pose, synced_opponents)

            radar_grid, points = preprocess_input(
                msg.pose.pose, synced_opponents, radar_window, bounding_boxes
            )

            ego_velocity = np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
            )

            velocities = preprocess_velocities(
                msg.pose.pose.orientation, ego_velocity, synced_opponents
            )

            save_frame_data(
                frame_id, bounding_boxes, radar_grid, points, velocities, output_dirs
            )

            frame_id += 1

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

    radar_window = deque()
    radar_counts = defaultdict(int)
    opponent_history = defaultdict(deque)

    frame_id = 0
    for topic, msg, timestamp in tqdm(reader.read_messages()):
        time_sec = timestamp * 1e-9
        frame_id = process_message(
            topic,
            msg,
            time_sec,
            radar_window,
            radar_counts,
            opponent_history,
            frame_id,
            output_dirs,
        )


if __name__ == "__main__":
    main()
