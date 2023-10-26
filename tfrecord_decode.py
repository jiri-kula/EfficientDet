#%%
import tensorflow as tf

import numpy as np
import IPython.display as display
import cv2 as cv
import pandas as pd
from dataset_api import load_image, column_names
from progress.bar import Bar

example_path = "zaznamy_z_vyroby.tfrecord"

# Read the data back out.
def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      {
        "image_raw": tf.io.FixedLenFeature([], dtype=tf.string),
        "xmins": tf.io.VarLenFeature(dtype=tf.float32),
        "ymins": tf.io.VarLenFeature(dtype=tf.float32),
        "xmaxes": tf.io.VarLenFeature(dtype=tf.float32),
        "xmaxes": tf.io.VarLenFeature(dtype=tf.float32),
        "r11s": tf.io.VarLenFeature(dtype=tf.float32),
        "r21s": tf.io.VarLenFeature(dtype=tf.float32),
        "r31s": tf.io.VarLenFeature(dtype=tf.float32),
        "r12s": tf.io.VarLenFeature(dtype=tf.float32),
        "r22s": tf.io.VarLenFeature(dtype=tf.float32),
        "r32s": tf.io.VarLenFeature(dtype=tf.float32)
      }
  )

# single = 0
# multi = 0
# for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
#     xmins = batch["xmins"]
#     if len(xmins.values) > 1:
#       multi += 1
#     else:
#       single += 1
#       # print(xmins.values)

# print("single", single)
# print("multi", multi)