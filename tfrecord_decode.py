# %%
import tensorflow as tf

import numpy as np
import IPython.display as display
import cv2 as cv
import pandas as pd
from dataset_api import load_image, column_names
from progress.bar import Bar
from model.anchors import SamplesEncoder

example_path = "zaznamy_z_vyroby.tfrecord"

IMG_OUT_SIZE = 320


def get_boxes(xmins, ymins, xmaxes, ymaxes):
    # N = len(item["xmins"])
    # x1 = np.zeros((N, 1))
    # x2 = np.zeros((N, 1))
    # y1 = np.zeros((N, 1))
    # y2 = np.zeros((N, 1))

    # for i in N:
    x1 = xmins * IMG_OUT_SIZE
    x2 = xmaxes * IMG_OUT_SIZE
    y1 = ymins * IMG_OUT_SIZE
    y2 = ymaxes * IMG_OUT_SIZE

    w = tf.subtract(x2, x1)
    h = tf.subtract(y2, y1)

    # where each box is of the format [x, y, width, height]
    # box = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h], dtype=np.float32)

    gt_boxes = tf.cast(
        tf.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h], axis=1), tf.float32
    )

    return gt_boxes


# Read the data back out.
# @tf.autograph.experimental.do_not_convert
@tf.function
def decode_raw(record_bytes):
    item = tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        {
            "image_path": tf.io.FixedLenFeature([], dtype=tf.string),
            "image_raw": tf.io.FixedLenFeature([], dtype=tf.string),
            "classes": tf.io.VarLenFeature(dtype=tf.float32),
            "xmins": tf.io.VarLenFeature(dtype=tf.float32),
            "ymins": tf.io.VarLenFeature(dtype=tf.float32),
            "xmaxes": tf.io.VarLenFeature(dtype=tf.float32),
            "ymaxes": tf.io.VarLenFeature(dtype=tf.float32),
            "r11s": tf.io.VarLenFeature(dtype=tf.float32),
            "r21s": tf.io.VarLenFeature(dtype=tf.float32),
            "r31s": tf.io.VarLenFeature(dtype=tf.float32),
            "r12s": tf.io.VarLenFeature(dtype=tf.float32),
            "r22s": tf.io.VarLenFeature(dtype=tf.float32),
            "r32s": tf.io.VarLenFeature(dtype=tf.float32),
        },
    )
    return item


def raw2label(item):
    # image
    image = tf.io.decode_png(item["image_raw"])
    image = tf.image.resize(image, [320, 320])

    # classes
    gt_classes = tf.sparse.to_dense(item["classes"], default_value=-1.0)
    gt_classes = tf.reshape(gt_classes, [-1, 1])

    # boxes
    xmins = tf.sparse.to_dense(item["xmins"], default_value=0.0)
    ymins = tf.sparse.to_dense(item["ymins"], default_value=0.0)
    xmaxes = tf.sparse.to_dense(item["xmaxes"], default_value=0.0)
    ymaxes = tf.sparse.to_dense(item["ymaxes"], default_value=0.0)
    gt_boxes = get_boxes(xmins, ymins, xmaxes, ymaxes)

    # angles
    r11 = tf.sparse.to_dense(item["r11s"], default_value=0.0)
    r21 = tf.sparse.to_dense(item["r21s"], default_value=0.0)
    r31 = tf.sparse.to_dense(item["r31s"], default_value=0.0)
    r12 = tf.sparse.to_dense(item["r12s"], default_value=0.0)
    r22 = tf.sparse.to_dense(item["r22s"], default_value=0.0)
    r32 = tf.sparse.to_dense(item["r32s"], default_value=0.0)

    gt_angles = tf.stack([r11, r21, r31, r12, r22, r32])

    gt_angles = tf.reshape(gt_angles, [-1, 6])

    return image, gt_boxes, gt_classes, gt_angles


# @tf.function
@tf.autograph.experimental.do_not_convert
def decode_fn(record_bytes):
    item = decode_raw(record_bytes)

    image, gt_boxes, gt_classes, gt_angles = raw2label(item)

    se = SamplesEncoder()
    label = se._encode_sample(image.shape, gt_boxes, gt_classes, gt_angles)

    return image, label


if __name__ == "__main__":
    for batch in tf.data.TFRecordDataset([example_path]):  # .map(decode_fn):
        example = tf.train.Example()
        example.ParseFromString(batch.numpy())
        print(example)
