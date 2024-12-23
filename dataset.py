import keras
import numpy as np
import tensorflow as tf
from model.anchors import SamplesEncoder
from model.utils import resize_and_pad
import os
import json
import cv2 as cv
from imgaug import augmenters as iaa
import csv
from enum import IntEnum
from dataclasses import dataclass
from typing import List
from progress.bar import Bar

IMG_FILE_SIZE = 320.0
IMG_OUT_SIZE = 320.0
IMG_SCALE = IMG_OUT_SIZE / IMG_FILE_SIZE


def get_image_names(data_dir):
    for root_dir, cur_dir, files in os.walk(data_dir):
        return [file for file in files if os.path.splitext(file)[1] == ".png"]


def is_box(val):
    return val["shape_type"] == "rectangle"


def is_box_point(val, groupid):
    return val["shape_type"] == "point" and val["group_id"] == groupid


def labelme_to_coco(labelme_data):
    shapes = labelme_data["shapes"]
    image_width = labelme_data["imageWidth"]
    image_height = labelme_data["imageHeight"]
    bboxes = [shape for shape in shapes if is_box(shape)]

    # out_classes = []  # group == class_id

    objs = []
    for bbox in bboxes:
        out_points = []
        group_id = bbox["group_id"]

        x1, y1 = bbox["points"][0]
        x2, y2 = bbox["points"][1]

        if x1 > x2:
            tmp = x2
            x2 = x1
            x1 = tmp

        if y1 > y2:
            tmp = y2
            y2 = y1
            y1 = tmp

        # int_box = [
        #     int(item) for item in [x1, y1, x2 - x1, y2 - y1]
        # ]  # convert to integers

        box_keypoints = [shape for shape in shapes if is_box_point(shape, group_id)]
        box_keypoints = sorted(box_keypoints, key=lambda x: x["label"])

        my_points = []
        for point in box_keypoints:
            my_point = [int(item) for item in point["points"][0]]
            my_point.append(1)
            my_points.append(my_point)

        out_points.append(my_points)

        w = x2 - x1
        h = y2 - y1
        objs.append(
            {
                "class_id": group_id - 1,
                "box": np.array([x1 - w / 2.0, y1 - h / 2.0, w, h], dtype=np.float32),
                # "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                "keypoints": out_points,
            }
        )

    # coco_data = {"classes": out_classes, {"bboxes": out_boxes, "keypoints": out_points}}

    return objs


def image_mosaic(images, delay=1):
    N = len(images)

    if N == 0:
        return

    R = int(np.floor(np.sqrt(N)))
    C = int(N / R)

    THUMB_SIZE = 128
    img = np.zeros(
        (R * THUMB_SIZE, C * THUMB_SIZE, 3), dtype=np.uint8
    )  # notice the switch of R, C here ...

    r = 0
    c = 0

    M = np.eye(2, 3)

    M[0][0] = M[1][1] = THUMB_SIZE / images[0].shape[0]

    for image in images:
        u = np.uint8(image)
        # cv.imshow("batch_u", u)
        # cv.waitKey()

        # ... and here as numpy users row major order, while cv size uses (width, height) order
        cv.warpAffine(
            u, M, (C * THUMB_SIZE, R * THUMB_SIZE), img, 0, cv.BORDER_TRANSPARENT
        )

        c += 1
        if c == C:
            c = 0
            r += 1

        M[0][2] = THUMB_SIZE * c
        M[1][2] = THUMB_SIZE * r

    bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow("batch", bgr)
    cv.waitKey(delay)
    # cv.destroyAllWindows()


# CSV
class IDX(IntEnum):
    PURPOSE = 0
    PATH = 1
    OBJECT = 2
    X1 = 3
    Y1 = 4
    X2 = 7
    Y2 = 8
    R11 = 9
    R21 = 10
    R31 = 11
    R12 = 12
    R22 = 13
    R32 = 14


def unique_paths(reader, file, purpose=None):
    unique = []

    file.seek(0)
    for row in reader:
        accept = True
        if purpose is not None:
            accept = row[IDX.PURPOSE] == purpose
        if accept and row[IDX.PATH] not in unique:
            unique.append(row[IDX.PATH])

    return unique


labels_map = {
    "drazka_rv5": 0,
    "nedrazka_rv5": 1,
    "drazka_rv8": 2,
    "nedrazka_rv8": 3,
    "drazka_rv12": 4,
    "nedrazka_rv12": 5,
}


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    lbl: int
    r11: float
    r21: float
    r31: float
    r21: float
    r22: float
    r32: float

    def __init__(self, row):
        x1 = float(row[IDX.X1])
        y1 = float(row[IDX.Y1])
        x2 = float(row[IDX.X2])
        y2 = float(row[IDX.Y2])

        if x1 > x2:
            x1, x2 = x2, x1

        if y1 > y2:
            y1, y2 = y2, y1

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.lbl = labels_map[row[IDX.OBJECT]]
        self.r11 = float(row[IDX.R11])
        self.r21 = float(row[IDX.R21])
        self.r31 = float(row[IDX.R31])
        self.r12 = float(row[IDX.R12])
        self.r22 = float(row[IDX.R22])
        self.r32 = float(row[IDX.R32])

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def area(self):
        return self.width() * self.height()


@dataclass
class Sample:
    # image_path: str
    boxes: list

    def __init__(self, row):
        # self.image_path = row[IDX.PATH]
        self.boxes = []

        self.boxes.append(Box(row))


class CSVDataset(keras.utils.all_utils.Sequence):
    def __init__(self, meta_path, aug, batch_size, train=True):
        self.batch_size = batch_size

        with open(meta_path, "r") as infile:
            reader = csv.reader(infile, lineterminator="\n", quoting=csv.QUOTE_NONE)

            num_rows = 0
            for row in reader:
                num_rows += 1

            self.samples = dict()

            bar = Bar(
                "Build cache",
                max=num_rows,
                suffix="%(percent).1f%% - %(eta)ds",
            )

            infile.seek(0)
            for row in reader:
                # samples = [_ for _ in self.samples if _.image_path == row[IDX.PATH]]
                key = row[IDX.PATH]
                sample = self.samples.get(key)
                if sample is None:
                    self.samples[key] = Sample(row)
                else:
                    sample.boxes.append(Box(row))
                bar.next()
            bar.finish()

            self.samples = list(self.samples.items())

        self.se = SamplesEncoder()

        self.batch = [None] * self.__len__()

    def __len__(self):
        return len(self.samples) // self.batch_size
        # return 1

    def __getitem__(self, index):
        # if self.batch[index] is None:
        #     self.batch[index] = self.get_batch(index)

        # return self.batch[index]
        return self.get_batch(index)

    def get_batch(self, index):
        train_images = []
        lbl_boxes = []
        lbl_classes = []
        lbl_angles = []

        idx_from = index * self.batch_size
        idx_to = min(idx_from + self.batch_size, len(self.samples))

        for idx in range(idx_from, idx_to):
            sample = self.samples[idx]

            image_path = sample[0]
            boxes = sample[1].boxes

            Image = tf.keras.utils.load_img(image_path)
            Image = Image.resize((IMG_OUT_SIZE, IMG_OUT_SIZE))
            Image = tf.keras.utils.img_to_array(
                Image, dtype=np.float32
            )  # see inference.py / 255.0

            train_images.append(Image)

            # boxes
            num_boxes = len(boxes)
            BoundingBoxes = np.zeros((num_boxes, 4), dtype=np.float32)
            Classes = np.zeros((num_boxes,), dtype=np.float32)
            Angles = np.zeros((num_boxes, 6), dtype=np.float32)

            for iobj, obj in enumerate(boxes):
                x1 = obj.x1 * IMG_OUT_SIZE
                x2 = obj.x2 * IMG_OUT_SIZE
                y1 = obj.y1 * IMG_OUT_SIZE
                y2 = obj.y2 * IMG_OUT_SIZE

                if not (x2 > x1 and y2 > y1):
                    raise ValueError(
                        "assertion failed in {:s}: x1: {:f}, x2: {:f} and y1: {:f}, y2: {:f}".format(
                            image_path, x1, x2, y1, y2
                        )
                    )

                w = x2 - x1
                h = y2 - y1

                # where each box is of the format [x, y, width, height]
                box = np.array(
                    [(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h], dtype=np.float32
                )

                BoundingBoxes[iobj] = box
                Classes[iobj] = float(obj.lbl)
                Angles[iobj][0] = float(obj.r11)
                Angles[iobj][1] = float(obj.r21)
                Angles[iobj][2] = float(obj.r31)
                Angles[iobj][3] = float(obj.r12)
                Angles[iobj][4] = float(obj.r22)
                Angles[iobj][5] = float(obj.r32)

            lbl_boxes.append(BoundingBoxes)
            lbl_classes.append(Classes)
            lbl_angles.append(Angles)

        if (
            not (len(lbl_boxes) == len(lbl_classes) == len(train_images))
            or len(train_images) < 1
        ):
            raise ValueError(
                "Unexpected dimension of MyDataset. The size of boxes: {:d}, classes: {:d} and images: {:d}; batch index: {:d}, files from: {:d}, to: {:d}".format(
                    len(lbl_boxes),
                    len(lbl_classes),
                    len(train_images),
                    index,
                    idx_from,
                    idx_to,
                )
            )

        # train_images_aug = self.seq(images=train_images)

        retval = self.se.encode_batch(
            np.array(train_images), lbl_boxes, lbl_classes, lbl_angles
        )

        assert retval[1].shape[0] == self.batch_size

        return retval
