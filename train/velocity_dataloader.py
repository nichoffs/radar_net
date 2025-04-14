import os

import numpy as np
from torch.utils.data import Dataset


class VelocityDataset(Dataset):
    def __init__(self, points_path, labels_path):
        self.points_path = points_path
        self.labels_path = labels_path

    # def __len__(self):


max_num_points = 0
min_num_points = 0
for root, d, files in os.walk("/mnt/hdd/adjusted/points"):
    for file in files:
        num_points = np.load(os.path.join(root, file)).shape[0]
        max_num_points = max(max_num_points, num_points)
        min_num_points = min(min_num_points, num_points)
print(max_num_points)
print(min_num_points)
