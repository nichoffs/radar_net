from math import atan2, cos, sin

import cv2
import numpy as np

from utils import timestamp_to_sec

MAX_ANGLE = 9.8648575
MIN_ANGLE = -np.pi / 2
MAX_RANGE = 327.675
MIN_RANGE = 0

SENSOR_POSITIONS = {
    "front": (1.789, 0.0, 0.578),
    "rear": (-0.752, 0.0, 0.107),
}

GRID_SIZE = 800
METER_RANGE = 200
PIXEL_RESOLUTION = METER_RANGE / GRID_SIZE

LENGTH = 4.91
WIDTH = 1.905

VEHICLE_CORNERS = np.array(
    [
        [LENGTH / 2, WIDTH / 2],
        [LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, WIDTH / 2],
    ]
)


def compute_relative_translation(ego_pose, opp_pose, ego_yaw):
    dx_global = opp_pose.position.x - ego_pose.position.x
    dy_global = opp_pose.position.y - ego_pose.position.y
    dx_rel = dx_global * cos(ego_yaw) + dy_global * sin(ego_yaw)
    dy_rel = -dx_global * sin(ego_yaw) + dy_global * cos(ego_yaw)
    return dx_rel, dy_rel


def create_transformation_matrix(rel_yaw, translation):
    tx, ty = translation
    return np.array(
        [[cos(rel_yaw), -sin(rel_yaw), tx], [sin(rel_yaw), cos(rel_yaw), ty], [0, 0, 1]]
    )


def transform_corners(transform, corners):
    num_corners = corners.shape[0]
    corners_homogeneous = np.hstack([corners, np.ones((num_corners, 1))])
    transformed = (transform @ corners_homogeneous.T).T
    return transformed[:, :2]


def normalize_field(raw_values, norm_min, norm_max, raw_max=65535.0):
    return (norm_max - norm_min) * (raw_values / raw_max) + norm_min


def compute_sensor_coordinates(
    anchor, ranges, sin_elev, azimuth_angles, sensor_is_rear
):
    direction_multiplier = -1 if sensor_is_rear else 1
    if sensor_is_rear:
        offset = np.radians(1)
        x = anchor[0] + direction_multiplier * ranges * sin_elev * np.cos(
            azimuth_angles + offset
        )
        y = anchor[1] + direction_multiplier * ranges * sin_elev * np.sin(
            azimuth_angles + offset
        )
    else:
        x = anchor[0] + direction_multiplier * ranges * sin_elev * np.cos(
            azimuth_angles
        )
        y = anchor[1] + direction_multiplier * ranges * sin_elev * np.sin(
            azimuth_angles
        )
    return x, y


def compute_normalized_timestamp(
    curr_timestamp, global_min_time, global_time_range, num_points
):
    if global_time_range < 1e-9:
        return np.zeros(num_points, dtype=np.float32)
    norm_val = (curr_timestamp - global_min_time) / global_time_range
    return np.full(num_points, norm_val, dtype=np.float32)


def process_radar_message(radar, global_min_time, global_time_range):
    msg = radar.msg
    sensor_is_rear = radar.sensor_type == "rear"
    anchor = np.array(SENSOR_POSITIONS["rear" if sensor_is_rear else "front"])
    doppler_raw = np.array(
        [p.radial_velocity for p in msg.detections], dtype=np.float32
    )
    elev_raw = np.array([p.elevation_angle for p in msg.detections], dtype=np.float32)
    rcs_raw = np.array(
        [p.radar_cross_section for p in msg.detections], dtype=np.float32
    )
    azimuth_raw = np.array([p.azimuth_angle for p in msg.detections], dtype=np.float32)
    range_raw = np.array([p.range for p in msg.detections], dtype=np.float32)
    doppler_norm = doppler_raw / 65535.0
    rcs_norm = rcs_raw / 65535.0
    elevation_angles = normalize_field(elev_raw, MIN_ANGLE, MAX_ANGLE)
    azimuth_angles = normalize_field(azimuth_raw, MIN_ANGLE, MAX_ANGLE)
    ranges_norm = normalize_field(range_raw, MIN_RANGE, MAX_RANGE)
    sin_elev = np.sin(np.pi / 2 - elevation_angles)
    x, y = compute_sensor_coordinates(
        anchor, ranges_norm, sin_elev, azimuth_angles, sensor_is_rear
    )
    curr_timestamp = timestamp_to_sec(msg)
    norm_timestamps = compute_normalized_timestamp(
        curr_timestamp, global_min_time, global_time_range, len(doppler_raw)
    )
    points = np.column_stack([doppler_norm, rcs_norm, norm_timestamps, x, y])
    return points


def normalize_radar_detections(radar_window):
    if not radar_window:
        return np.zeros((0, 5), dtype=np.float32)
    timestamps = np.array([timestamp_to_sec(radar.msg) for radar in radar_window])
    global_min_time = timestamps.min()
    global_time_range = (
        timestamps.max() - global_min_time if len(timestamps) > 1 else 1e-9
    )
    all_points = []
    for radar in radar_window:
        points = process_radar_message(radar, global_min_time, global_time_range)
        all_points.append(points)
    return np.vstack(all_points)


def positions_to_grid_indices(positions, grid_size, meter_range):
    pixel_resolution = meter_range / grid_size
    x_shifted = positions[:, 0] + meter_range / 2
    y_shifted = positions[:, 1] + meter_range / 2
    x_indices = (x_shifted / pixel_resolution).astype(int)
    y_indices = (y_shifted / pixel_resolution).astype(int)
    return x_indices, y_indices


def map_points_to_grid(
    normalized_detections, grid_size=GRID_SIZE, meter_range=METER_RANGE
):
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    count_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    features = normalized_detections[:, :3]
    positions = normalized_detections[:, 3:5]

    x_indices, y_indices = positions_to_grid_indices(positions, grid_size, meter_range)

    valid_mask = (
        (x_indices >= 0)
        & (x_indices < grid_size)
        & (y_indices >= 0)
        & (y_indices < grid_size)
    )
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    valid_features = features[valid_mask]

    for i in range(len(x_indices)):
        xx = x_indices[i]
        yy = y_indices[i]
        curr_count = count_grid[yy, xx]
        grid[yy, xx] = (grid[yy, xx] * curr_count + valid_features[i]) / (
            curr_count + 1
        )
        count_grid[yy, xx] += 1

    return grid


def point_in_polygon(point, polygon):
    poly = polygon.reshape((-1, 1, 2)).astype(np.float32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def get_points_in_bboxes(vehicle_boxes, normalized_detections):
    result = {vid: [] for vid, _ in vehicle_boxes}
    for detection in normalized_detections:
        point_xy = tuple(detection[3:5])
        point_feat = detection[:3]
        for vehicle_id, box in vehicle_boxes:
            if point_in_polygon(point_xy, box):
                result[vehicle_id].append(point_feat)
                break
    for vehicle_id in result:
        result[vehicle_id] = np.array(result[vehicle_id], dtype=np.float32)
    return result


def compute_relative_velocity(ego_yaw, ego_velocity, opp_velocity):
    v_rel_map = opp_velocity - ego_velocity
    R_map_to_ego = np.array(
        [[cos(-ego_yaw), -sin(-ego_yaw)], [sin(-ego_yaw), cos(-ego_yaw)]]
    )
    v_rel_ego = R_map_to_ego @ v_rel_map
    return v_rel_ego


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return atan2(siny_cosp, cosy_cosp)


def preprocess_per_vehicle(ego_pose, ego_velocity, synced_opponents, radar_window):
    ego_yaw = yaw_from_quaternion(ego_pose.orientation)

    normalized_detections = normalize_radar_detections(radar_window)
    radar_grid = map_points_to_grid(normalized_detections)

    vehicle_data = {}

    for vehicle_id, (opp_pose, vx, vy) in synced_opponents.items():
        opp_yaw = yaw_from_quaternion(opp_pose.orientation)
        rel_yaw = opp_yaw - ego_yaw
        translation = compute_relative_translation(ego_pose, opp_pose, ego_yaw)
        transform = create_transformation_matrix(rel_yaw, translation)
        box = transform_corners(transform, VEHICLE_CORNERS)

        if not np.any(np.linalg.norm(box, axis=-1) < 100):
            continue

        opp_velocity = np.array([vx, vy])
        rel_velocity = compute_relative_velocity(ego_yaw, ego_velocity, opp_velocity)

        points = []
        for detection in normalized_detections:
            if point_in_polygon(tuple(detection[3:5]), box):
                points.append(detection[:3])

        if points:
            vehicle_data[vehicle_id] = {
                "box": box,
                "velocity": rel_velocity,
                "points": np.array(points, dtype=np.float32),
            }

    return radar_grid, vehicle_data
