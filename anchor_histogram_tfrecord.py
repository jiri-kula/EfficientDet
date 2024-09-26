# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds


# %%
def analyze_bounding_boxes(dataset_name, split="train"):
    # Load the dataset
    ds = tfds.load(dataset_name, split=split, shuffle_files=False)

    # Initialize min and max bounding box dimensions
    min_bbox = [float("inf"), float("inf")]
    max_bbox = [float("-inf"), float("-inf")]

    # Iterate over the dataset
    for example in ds:
        objects = example["objects"]
        bboxes = objects["bbox"]

        for bbox in bboxes:
            ymin, xmin, ymax, xmax = bbox.numpy()
            width = xmax - xmin
            height = ymax - ymin

            # Update min and max dimensions
            if width < min_bbox[0]:
                min_bbox[0] = width
            if height < min_bbox[1]:
                min_bbox[1] = height
            if width > max_bbox[0]:
                max_bbox[0] = width
            if height > max_bbox[1]:
                max_bbox[1] = height

    # Output the results
    print(f"Minimum bounding box dimensions: {min_bbox}")
    print(f"Maximum bounding box dimensions: {max_bbox}")


# Example usage
analyze_bounding_boxes("kk_dataset:1.0.1")

# %%
