from math import atan2, cos, sin

import numpy as np

# Constants
LENGTH = 4.91  # Length of the vehicle in meters
WIDTH = 1.905  # Width of the vehicle in meters

# Vehicle corner coordinates (rectangle centered at origin)
VEHICLE_CORNERS = np.array(
    [
        [LENGTH / 2, WIDTH / 2],
        [LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, WIDTH / 2],
    ]
)


def calculate_yaw(qw, qx, qy, qz):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    return atan2(siny_cosp, cosy_cosp)


def preprocess_output(ego_pose, opponent_poses):
    ego_x = ego_pose.position.x
    ego_y = ego_pose.position.y
    ego_yaw = calculate_yaw(
        ego_pose.orientation.w,
        ego_pose.orientation.x,
        ego_pose.orientation.y,
        ego_pose.orientation.z,
    )

    transformed_corners = []
    for vehicle_id, opp_pose in opponent_poses.items():
        opp_yaw = calculate_yaw(
            opp_pose.orientation.w,
            opp_pose.orientation.x,
            opp_pose.orientation.y,
            opp_pose.orientation.z,
        )
        rel_yaw = opp_yaw - ego_yaw

        dx_global = opp_pose.position.x - ego_x
        dy_global = opp_pose.position.y - ego_y

        # Transform to ego-relative coordinates
        dx_rel = dx_global * cos(ego_yaw) + dy_global * sin(ego_yaw)
        dy_rel = -dx_global * sin(ego_yaw) + dy_global * cos(ego_yaw)

        # Create transformation matrix for the opponent vehicle
        transform = np.array(
            [
                [cos(rel_yaw), -sin(rel_yaw), dx_rel],
                [sin(rel_yaw), cos(rel_yaw), dy_rel],
                [0, 0, 1],
            ]
        )

        corners_homogeneous = np.hstack(
            [VEHICLE_CORNERS, np.ones((len(VEHICLE_CORNERS), 1))]
        )
        transformed_corners.append((transform @ corners_homogeneous.T).T[:, :2])

    return np.stack(transformed_corners, axis=0)
