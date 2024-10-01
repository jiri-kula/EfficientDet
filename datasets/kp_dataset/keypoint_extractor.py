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

resize_w = 224
resize_h = resize_w

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

    s = 2 * max(w, h)

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

    positive_roi = (lt[0], lt[1], rb[0], rb[1])

    return M, positive_roi


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

        objects.append({"bbox": bbox})

    yield src_filepath, warped, objects

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
