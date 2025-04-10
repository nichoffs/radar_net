import os

import cv2
import yaml

YAML_PTH = "/home/nic/code/radar_net/data/dataset.yaml"

with open(YAML_PTH, "r") as f:
    config = yaml.safe_load(f)

DS_PATH = config["path"]

METERS_PER_PIXEL = 0.25

# Final dataset should have shape (T, NUM_POINTS_INSIDE, 3)

# How to do this?

# Iterate through images, bounding box labels, and velocities in DS_PATH
# Get points inside bounding box using labels
# Append points to dataset to serve as input

root_images, _, images = next(os.walk(DS_PATH + "/images"))
root_labels, _, labels = next(os.walk(DS_PATH + "/labels"))
root_velocities, _, velocities = next(os.walk(DS_PATH + "/velocities"))

NUM_FRAMES = len(images)

for i in range(NUM_FRAMES):
    image_path = os.path.join(root_images, images[i])
    label_path = os.path.join(root_labels, labels[i])
    velocity_path = os.path.join(root_velocities, velocities[i])

    image = cv2.imread(image_path)
    with open(velocity_path, "r") as f:
        velocity = float(f.readline().split()[1])

    with open(label_path, "r") as f:
        label = f.readline()

    print(label_path)
    print(label)
    break
