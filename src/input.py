import numpy as np

from utils import timestamp_to_sec

# Constants for normalization
MAX_ANGLE = 9.8648575
MIN_ANGLE = -np.pi / 2
MAX_RANGE = 327.675
MIN_RANGE = 0

SENSOR_POSITIONS = {"front": (1.789, 0.0, 0.578), "rear": (-0.752, 0.0, 0.107)}

GRID_SIZE = 800
METER_RANGE = 200
PIXEL_RESOLUTION = METER_RANGE / GRID_SIZE


def normalize_radar_detections(radar_window):
    # Get timestamps for temporal normalization
    timestamps = np.array([timestamp_to_sec(radar.msg) for radar in radar_window])
    min_time = timestamps.min()
    time_range = timestamps.max() - min_time

    # Pre-calculate lists for batch processing
    all_points = []

    # Gather all points and their metadata
    for radar in radar_window:
        msg = radar.msg
        is_rear = radar.sensor_type == "rear"
        m = -1 if is_rear else 1
        anchor = np.array(
            SENSOR_POSITIONS["rear"] if is_rear else SENSOR_POSITIONS["front"]
        )

        # Extract features into numpy arrays
        doppler = np.array([p.radial_velocity for p in msg.detections])
        elevations = np.array([p.elevation_angle for p in msg.detections])
        rcs = np.array([p.radar_cross_section for p in msg.detections])
        azimuths = np.array([p.azimuth_angle for p in msg.detections])
        ranges = np.array([p.range for p in msg.detections])

        # Normalize raw features to [0, 1]
        features = np.column_stack([doppler / 65535, rcs / 65535])

        # Calculate actual angles
        elevation_angles = (MAX_ANGLE - MIN_ANGLE) * (elevations / 65535) + MIN_ANGLE
        azimuth_angles = (MAX_ANGLE - MIN_ANGLE) * (azimuths / 65535) + MIN_ANGLE
        ranges = (MAX_RANGE - MIN_RANGE) * (ranges / 65535) + MIN_RANGE

        # Calculate positions
        sin_elev = np.sin(np.pi / 2 - elevation_angles)
        if m == -1:
            x = anchor[0] + m * ranges * sin_elev * np.cos(
                azimuth_angles + np.radians(1)
            )
            y = anchor[1] + m * ranges * sin_elev * np.sin(
                azimuth_angles + np.radians(1)
            )
        else:
            x = anchor[0] + m * ranges * sin_elev * np.cos(azimuth_angles)
            y = anchor[1] + m * ranges * sin_elev * np.sin(azimuth_angles)

        # Add temporal feature
        curr_timestamp = timestamp_to_sec(msg)
        if time_range == 0:
            norm_timestamps = np.zeros(msg.number_detections)
        else:
            norm_timestamps = np.full(
                msg.number_detections, (curr_timestamp - min_time) / time_range
            )

        # Combine all features and coordinates
        points = np.column_stack([features, norm_timestamps, x, y])
        all_points.append(points)

    return np.vstack(all_points)


def map_points_to_grid(normalized_detections, grid_size=800, meter_range=200):
    grid = np.zeros((grid_size, grid_size, 3))
    count_grid = np.zeros((grid_size, grid_size))
    pixel_resolution = meter_range / grid_size

    features = normalized_detections[:, :3]  # All 3 features including timestamp
    positions = normalized_detections[:, 3:]  # x, y coordinates

    # Shift coordinates to make (0,0) at the bottom-left of the grid
    x_shifted = positions[:, 0] + meter_range / 2
    y_shifted = positions[:, 1] + meter_range / 2

    x_indices = (x_shifted / pixel_resolution).astype(int)
    y_indices = (y_shifted / pixel_resolution).astype(int)

    # Filter out points outside the grid
    valid_mask = (
        (x_indices >= 0)
        & (x_indices < grid_size)
        & (y_indices >= 0)
        & (y_indices < grid_size)
    )

    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    valid_features = features[valid_mask]

    # Group points by grid cell
    for i in range(len(x_indices)):
        x_idx = x_indices[i]
        y_idx = y_indices[i]

        # Update all feature channels (including timestamp) with running average
        current_count = count_grid[y_idx, x_idx]
        if current_count == 0:
            grid[y_idx, x_idx] = valid_features[i]
        else:
            # Compute running average for all features including timestamp
            grid[y_idx, x_idx] = (
                grid[y_idx, x_idx] * current_count + valid_features[i]
            ) / (current_count + 1)

        # Increment count for averaging
        count_grid[y_idx, x_idx] += 1

    return grid


def preprocess_input(ego_odom, radar_window):
    # Get normalized detections
    normalized_detections = normalize_radar_detections(radar_window)
    # Map points to grid
    grid = map_points_to_grid(normalized_detections)
    return grid
