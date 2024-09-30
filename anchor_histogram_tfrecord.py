# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


# %%
def analyze_bounding_boxes(ds):
    # Initialize min and max bounding box dimensions
    min_bbox = [float("inf"), float("inf")]
    max_bbox = [float("-inf"), float("-inf")]

    # Iterate over the dataset
    for i, example in enumerate(tqdm(ds)):
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
    print(f"Minimum bounding box dimensions: {320 * np.array(min_bbox)}")
    print(f"Maximum bounding box dimensions: {320 * np.array(max_bbox)}")


# Example usage

ds0 = tfds.load("kk_dataset:1.0.0", split="train", shuffle_files=True)
ds1 = tfds.load("kk_dataset:1.0.1", split="train", shuffle_files=True)
ds2 = tfds.load("kk_dataset:1.0.2", split="train", shuffle_files=True)
ds = ds0.concatenate(ds1.concatenate(ds2))

analyze_bounding_boxes(ds0)

# %%
