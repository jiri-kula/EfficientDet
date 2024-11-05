# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tqdm import tqdm

from datasets.decode_function import concat_datasets
from model.anchors import INPUT_SIZE

import tensorflow_datasets as tfds


# %%
def analyze_bounding_boxes(ds):
    # Initialize min and max bounding box dimensions
    min_bbox = [float("inf"), float("inf")]
    max_bbox = [float("-inf"), float("-inf")]

    # Initialize min and max aspect ratios
    min_aspect_ratio = float("inf")
    max_aspect_ratio = float("-inf")

    # Iterate over the dataset
    for i, example in enumerate(tqdm(ds)):
        objects = example["objects"]
        bboxes = objects["bbox"]

        for bbox in bboxes:
            ymin, xmin, ymax, xmax = bbox.numpy()
            width = xmax - xmin
            height = ymax - ymin

            # Calculate aspect ratio
            aspect_ratio = width / height

            # Update min and max dimensions
            if width < min_bbox[0]:
                min_bbox[0] = width
            if height < min_bbox[1]:
                min_bbox[1] = height
            if width > max_bbox[0]:
                max_bbox[0] = width
            if height > max_bbox[1]:
                max_bbox[1] = height

            # Update min and max aspect ratios
            if aspect_ratio < min_aspect_ratio:
                min_aspect_ratio = aspect_ratio
            if aspect_ratio > max_aspect_ratio:
                max_aspect_ratio = aspect_ratio

    # Output the results
    min_bbox = INPUT_SIZE * np.array(min_bbox)
    max_bbox = INPUT_SIZE * np.array(max_bbox)
    print(
        "Minimum bounding box width: {:.2f}, height: {:.2f}".format(
            min_bbox[0], min_bbox[1]
        )
    )
    print(
        "Maximum bounding box width: {:.2f}, height {:.2f}".format(
            max_bbox[0], max_bbox[1]
        )
    )
    print("Minimum aspect ratio: {:.2f}".format(min_aspect_ratio))
    print("Maximum aspect ratio: {:.2f}".format(max_aspect_ratio))


# Example usage

analyze_bounding_boxes(concat_datasets())

# %%
