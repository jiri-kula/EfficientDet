# %%
import tensorflow as tf

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

from tfrecord_decode import decode_raw, decode_fn, raw2label
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from mpl_toolkits.axes_grid1 import ImageGrid
from model.anchors import Anchors
import cv2 as cv


# %%
example_path = "/home/jiri/winpart/Edwards/zaznamy_z_vyroby.tfrecord"
# example_path = "/home/jiri/winpart/Edwards/merge-e.tfrecord"

ds = tf.data.TFRecordDataset([example_path]).skip(5000).take(1)

record = decode_raw(next(iter(ds)))
label = raw2label(record)

ds = tf.data.TFRecordDataset([example_path]).skip(0).take(1).map(decode_fn)


# %%
img = sample[0]
lbl = sample[1]

# %%
plt.imshow(tf.cast(img, tf.uint8))
uimg = tf.cast(img, tf.uint8)
cv.imwrite("tmp.png", uimg.numpy())

# %%
pos = tf.squeeze(tf.where(lbl[:, 10] > -1.0))

lbl_encoding = tf.gather(lbl[:, :4], pos)

gen = Anchors()
a = gen.get_anchors(320, 320)
anchor_boxes = tf.gather(a, pos)

box_variance = tf.cast([0.1, 0.1, 0.2, 0.2], tf.float32)

boxes = tf.concat(
    [
        lbl_encoding[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
        tf.exp(lbl_encoding[..., 2:]) * anchor_boxes[..., 2:],
    ],
    axis=-1,
)

cls = tf.gather(lbl[:, 10], pos)


# boxes = to_corners(boxes)

# %%
