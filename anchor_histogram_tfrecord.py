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

    # Average rectangle
    avg_width = 0
    avg_height = 0
    avg_total = 0

    max_image = None

    # Iterate over the dataset
    for i, example in enumerate(tqdm(ds)):
        objects = example["objects"]
        bboxes = objects["bbox"]

        for bbox in bboxes:
            ymin, xmin, ymax, xmax = bbox.numpy()

            assert xmin >= 0.0
            assert ymin >= 0.0
            assert xmin <= 1.0
            assert ymin <= 1.0

            width = xmax - xmin
            height = ymax - ymin

            # Calculate aspect ratio
            aspect_ratio = width / height

            # Update min and max dimensions
            if width < min_bbox[0]:
                min_bbox[0] = width
                max_image = example["image"]
            if height < min_bbox[1]:
                min_bbox[1] = height
            if width > max_bbox[0]:
                max_bbox[0] = width
            if height > max_bbox[1]:
                max_bbox[1] = height

            avg_width += width
            avg_height += height
            avg_total += 1

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

    # Calculate average width and height
    avg_width *= INPUT_SIZE
    avg_height *= INPUT_SIZE
    avg_width /= avg_total
    avg_height /= avg_total

    print("Average width: {:.2f}".format(avg_width))
    print("Average height: {:.2f}".format(avg_height))

    plt.imshow(max_image)
    plt.show()


# Example usage

# analyze_bounding_boxes(concat_datasets())
# analyze_bounding_boxes(tfds.load("kk_dataset:1.2.3", split="train"))
analyze_bounding_boxes(tfds.load("kk_dataset:1.1.9", split="train"))

# %%
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import tensorflow_datasets as tfds
from tqdm import tqdm
from datasets.decode_function import concat_datasets


def suggest_anchor_boxes(dataset, num_clusters=9, input_size=320):
    """
    Suggest anchor boxes based on the training dataset.

    Args:
        dataset: A TensorFlow dataset containing bounding boxes.
        num_clusters: Number of anchor boxes to generate.
        input_size: Size of the input images.

    Returns:
        A list of suggested anchor boxes.
    """

    # First pass to count the number of bounding boxes
    # num_bboxes = 0
    # for data in tqdm(dataset, desc="Counting bounding boxes"):
    #     num_bboxes += len(data["objects"]["bbox"])

    # Preallocate the array for bounding boxes
    # bboxes = np.zeros((num_bboxes, 2))

    # Extract bounding boxes from the dataset
    bboxes = []
    # index = 0
    for data in tqdm(dataset, desc="Processing dataset"):
        for bbox in data["objects"]["bbox"]:
            ymin, xmin, ymax, xmax = bbox
            width = (xmax - xmin) * input_size
            height = (ymax - ymin) * input_size
            bboxes.append([width, height])
            # bboxes[index] = [width, height]
            # index += 1

    bboxes = np.array(bboxes)

    # Perform k-means clustering to find anchor boxes
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(bboxes)
    anchor_boxes = kmeans.cluster_centers_

    return anchor_boxes


import numpy as np


def calculate_anchor_box_parameters(clusters):
    """
    Calculate scales, aspect ratios, and areas from the provided clusters.

    Args:
        clusters: A list of anchor box clusters (width, height).

    Returns:
        scales: A list of scales for the anchor boxes.
        aspect_ratios: A list of aspect ratios for the anchor boxes.
        areas: A list of areas for the anchor boxes.
    """
    scales = []
    aspect_ratios = []
    areas = []

    for width, height in clusters:
        area = width * height
        scale = np.sqrt(area)
        aspect_ratio = width / height

        scales.append(scale)
        aspect_ratios.append(aspect_ratio)
        areas.append(area)

    return scales, aspect_ratios, np.sqrt(areas)


def select_representative_values(values, num_values=3):
    """
    Select representative values from a list.

    Args:
        values: A list of values.
        num_values: Number of representative values to select.

    Returns:
        A list of representative values.
    """
    sorted_values = sorted(values)
    step = len(sorted_values) // num_values
    representative_values = [sorted_values[i * step] for i in range(num_values)]
    return representative_values


# Example usage


# Example usage
dataset = tfds.load("kk_dataset:1.1.20", split="train")
# dataset = concat_datasets()

anchor_boxes = suggest_anchor_boxes(dataset)
# anchor_boxes = suggest_anchor_boxes(dataset.take(100))
print("Suggested anchor boxes (width, height):")
for box in anchor_boxes:
    print("Width: {:.2f}, Height: {:.2f}".format(box[0], box[1]))

scales, aspect_ratios, areas = calculate_anchor_box_parameters(anchor_boxes)
print("Scales:", scales)
print("Aspect Ratios:", aspect_ratios)
print("Areas:", areas)

# Select 3 representative scales and aspect ratios
representative_scales = select_representative_values(scales)
representative_scales = representative_scales / representative_scales[1]
representative_aspect_ratios = select_representative_values(aspect_ratios)
representative_areas = select_representative_values(areas)
print("Representative Scales:", representative_scales)
print("Representative Aspect Ratios:", representative_aspect_ratios)
print("Representative Areas:", representative_areas)
# %%
