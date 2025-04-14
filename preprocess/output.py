from math import cos, sin

import numpy as np

from utils import yaw_from_quaternion

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


def preprocess_bounding_boxes(ego_pose, opponent_poses):
    ego_yaw = yaw_from_quaternion(ego_pose.orientation)
    transformed_bboxes = []
    for vehicle_id, (opp_pose, v_x, v_y) in opponent_poses.items():
        opp_yaw = yaw_from_quaternion(opp_pose.orientation)
        rel_yaw = opp_yaw - ego_yaw
        translation = compute_relative_translation(ego_pose, opp_pose, ego_yaw)
        transform = create_transformation_matrix(rel_yaw, translation)
        transformed = transform_corners(transform, VEHICLE_CORNERS)
        transformed_bboxes.append((vehicle_id, transformed))
    return transformed_bboxes


def get_rotation_matrix(angle):
    return np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


def compute_relative_velocity(ego_yaw, ego_velocity, opp_velocity):
    v_rel_map = opp_velocity - ego_velocity
    R_map_to_ego = get_rotation_matrix(-ego_yaw)
    v_rel_ego = R_map_to_ego @ v_rel_map
    return v_rel_ego


def preprocess_velocities(ego_quat, ego_velocity, synced_opponents):
    ego_yaw = yaw_from_quaternion(ego_quat)
    relative_velocities = {}

    for vehicle_id, (_, velocity_x, velocity_y) in synced_opponents.items():
        opp_velocity = np.array([velocity_x, velocity_y])
        v_rel = compute_relative_velocity(ego_yaw, ego_velocity, opp_velocity)
        relative_velocities[vehicle_id] = v_rel

    return relative_velocities
