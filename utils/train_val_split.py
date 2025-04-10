#!/usr/bin/env python3
"""
This script creates a train/validation split from an existing dataset directory.
It does not modify the original dataset. Instead, it creates a new directory with the
specified name and copies over samples (images and their corresponding label files)
into the following structure:

    <output_path>/
      images/
        train/
        val/
      labels/
        train/
        val/

Each sample is assigned randomly to train or val based on the validation ratio.
"""

import argparse
import os
import random
import shutil


def create_dirs(output_path):
    """
    Creates the following directory structure under output_path:
      images/train, images/val, labels/train, labels/val.
    """
    dirs = [
        os.path.join(output_path, "images", "train"),
        os.path.join(output_path, "images", "val"),
        os.path.join(output_path, "labels", "train"),
        os.path.join(output_path, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Created directories under {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a train/val split from a dataset directory."
    )
    parser.add_argument(
        "-p",
        "--dataset-path",
        required=True,
        help="Path to the dataset directory containing 'images/' and 'labels/' subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Name (or path) of the output directory to create with the train/val split.",
    )
    parser.add_argument(
        "-ext",
        "--extension",
        default="png",
        help="Image file extension (default: png).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio (default: 0.2, meaning 20%% of the data will go to validation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output
    extension = args.extension
    val_ratio = args.val_ratio
    seed = args.seed

    # Ensure original dataset directories exist
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(
            f"Error: {dataset_path} must contain 'images/' and 'labels/' subdirectories."
        )
        return

    # Create output directory structure
    create_dirs(output_path)

    # List all image files with the provided extension in the images directory.
    all_images = [f for f in os.listdir(images_dir) if f.endswith(f".{extension}")]
    if not all_images:
        print(f"No image files with extension .{extension} found in {images_dir}.")
        return

    # For reproducibility, set the random seed
    random.seed(seed)

    # Shuffle the list
    random.shuffle(all_images)

    # Determine the split index
    n_total = len(all_images)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_files = all_images[:n_train]
    val_files = all_images[n_train:]

    print(f"Total samples: {n_total}")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

    # Function to copy both image and label file
    def copy_sample(file_name, subset):
        # file_name is something like "0.png"
        base_name = os.path.splitext(file_name)[0]
        src_image_path = os.path.join(images_dir, file_name)
        src_label_path = os.path.join(labels_dir, f"{base_name}.txt")

        dst_image_path = os.path.join(output_path, "images", subset, file_name)
        dst_label_path = os.path.join(output_path, "labels", subset, f"{base_name}.txt")

        # Copy the image file
        shutil.copy2(src_image_path, dst_image_path)
        # Check if the label file exists before copying.
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"Warning: Label file not found for image {file_name}")

    # Copy training samples
    for file_name in train_files:
        copy_sample(file_name, "train")

    # Copy validation samples
    for file_name in val_files:
        copy_sample(file_name, "val")

    print(f"Train/Val split completed. New dataset is located at: {output_path}")


if __name__ == "__main__":
    main()
