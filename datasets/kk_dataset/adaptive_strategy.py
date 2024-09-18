# Filename: default_strategy.py
# Created Date: 2024-09-12
# Modified Date: 2024-09-12
# Author: Jiri Kula
# Description:
# This file contains the default strategy for the KK dataset.
# The strategy is to crop the center square of the image and then resize the image to 320x320.

import csv, os
import cv2 as cv
import numpy as np
import tensorflow_datasets as tfds
import random

resize_w = 320
resize_h = resize_w


def get_intersection(roi1, roi2):
    # Unpack the ROIs
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    # Calculate the intersection rectangle
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = min(x1 + w1, x2 + w2) - x_inter
    h_inter = min(y1 + h1, y2 + h2) - y_inter

    # Check if there is an intersection
    if w_inter > 0 and h_inter > 0:
        return (x_inter, y_inter, w_inter, h_inter)
    else:
        return None


def get_random_non_overlapping_roi(image, existing_roi, roi_width, roi_height):
    img_height, img_width = image.shape[:2]
    x_min_existing, y_min_existing, x_max_existing, y_max_existing = existing_roi

    while True:
        # Generate random top-left corner for the new ROI
        x_min_new = random.randint(0, img_width - roi_width)
        y_min_new = random.randint(0, img_height - roi_height)
        x_max_new = x_min_new + roi_width
        y_max_new = y_min_new + roi_height

        if (
            get_intersection(
                (x_min_new, y_min_new, roi_width, roi_height),
                (
                    x_min_existing,
                    y_min_existing,
                    x_max_existing - x_min_existing,
                    y_max_existing - y_min_existing,
                ),
            )
            is None
        ):
            break

    # print(f"None overlapping ROI: {x_min_new}, {y_min_new}, {x_max_new}, {y_max_new}")

    return (x_min_new, y_min_new, x_max_new, y_max_new)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def calc_tform(image, xmins, ymins, xmaxs, ymaxs):
    height, width = image.shape[:2]

    x1 = min(xmins) * width
    y1 = min(ymins) * height
    x2 = max(xmaxs) * width
    y2 = max(ymaxs) * height

    w = x2 - x1
    h = y2 - y1

    s = max(w, h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    lt = np.array([cx - s / 2.0, cy - s / 2.0])
    rt = np.array([cx + s / 2.0, cy - s / 2.0])
    rb = np.array([cx + s / 2.0, cy + s / 2.0])

    # square roi in center of the image
    src = np.array([lt, rt, rb], dtype=np.float32)

    # destination square resized to destination image size
    dst = np.array([[0, 0], [resize_w, 0], [resize_w, resize_h]], dtype=np.float32)

    M = cv.getAffineTransform(src, dst)

    potivive_roi = (lt[0], lt[1], rb[0], rb[1])

    return M, potivive_roi


def store(clsnames, xmins, ymins, xmaxs, ymaxs, src_filepath):
    # load image from source file path
    image = cv.imread(src_filepath)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # calculate corp & resize transformation from source image to destination image

    tform, positive_roi = calc_tform(image, xmins, ymins, xmaxs, ymaxs)

    warped = cv.warpAffine(image, tform, (resize_w, resize_h), flags=cv.INTER_LINEAR)

    # calcuate new bouding box coordinates in the transormed destination image by reusing the transformation matrix of the source image
    height, width = image.shape[:2]
    topixels = np.array([[width, 0.0, 0.0], [0.0, height, 0.0], [0.0, 0.0, 1.0]])

    tonorm = np.array(
        [
            [1.0 / resize_w, 0.0],
            [0.0, 1.0 / resize_h],
        ]
    )

    # process individual bounding boxes
    objects = []

    for clsname, xmin, ymin, xmax, ymax in zip(clsnames, xmins, ymins, xmaxs, ymaxs):
        src_box_normalized = np.array([[xmin, ymin, 1], [xmax, ymax, 1]])  # 3x2 matrix

        dst_box = tonorm @ tform @ topixels @ src_box_normalized.T

        # build new row from filepath and box
        bbox = tfds.features.BBox(
            xmin=round(dst_box[0, 0], 4),
            ymin=round(dst_box[1, 0], 4),
            xmax=round(dst_box[0, 1], 4),
            ymax=round(dst_box[1, 1], 4),
        )

        objects.append({"bbox": bbox, "label": clsname})

    yield src_filepath, warped, objects

    negative_roi = get_random_non_overlapping_roi(
        image, positive_roi, resize_w, resize_h
    )

    negative_tform, _ = calc_tform(
        image,
        [negative_roi[0] / image.shape[1]],
        [negative_roi[1] / image.shape[0]],
        [negative_roi[2] / image.shape[1]],
        [negative_roi[3] / image.shape[0]],
    )
    neagative_warped = cv.warpAffine(
        image, negative_tform, (resize_w, resize_h), flags=cv.INTER_LINEAR
    )

    negative_objects = []
    yield src_filepath + "negative", neagative_warped, negative_objects


# import csv file and perform function on each row
def read_csv_file(source_ann_filepath):
    src_lastfilepath = None

    # center of adaptive bounding box
    purposes = []
    clsnames = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    with open(source_ann_filepath, "r") as file:
        reader = csv.reader(file)

        # last file path is None
        # read coordinates and file until file path is different or there are no more rows
        # if have coordinate - save them
        for row in reader:
            filepath = row[1]

            # applies for the first row of input file
            if src_lastfilepath is None:
                src_lastfilepath = filepath

            if filepath != src_lastfilepath:
                for image_path, image, objects in store(
                    clsnames, xmins, ymins, xmaxs, ymaxs, src_lastfilepath
                ):
                    yield image_path, image, objects

                # update
                src_lastfilepath = filepath

                # cleanup
                purposes = []
                clsnames = []
                xmins = []
                ymins = []
                xmaxs = []
                ymaxs = []

            # store this row
            purpose = row[0]
            clsname = row[2]
            xmin = float(row[3])
            ymin = float(row[4])
            xmax = float(row[7])
            ymax = float(row[8])

            purposes.append(purpose)
            clsnames.append(clsname)
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)

        # store last after all src rows were processed
        for image_path, image, objects in store(
            clsnames, xmins, ymins, xmaxs, ymaxs, src_lastfilepath
        ):
            yield image_path, image, objects
