# %%
EAGERLY = False

import sys

import tensorflow as tf

tf.config.run_functions_eagerly(EAGERLY)
if EAGERLY:
    tf.data.experimental.enable_debug_mode()

import pandas as pd
import numpy as np

from model.anchors import SamplesEncoder

sys.path.insert(1, "/home/jiri/DigitalAssistant/python")

from render_to_image import BladeGenerator


column_names = [
    "PURPOSE",
    "PATH",
    "OBJECT",
    "X1",
    "Y1",
    "UNUSED1",
    "UNUSED2",
    "X2",
    "Y2",
    "R11",
    "R21",
    "R31",
    "R12",
    "R22",
    "R32",
]


def load_image(image_path):
    raw_image = tf.io.read_file(image_path)
    image = tf.image.decode_png(raw_image, channels=3)
    image = tf.image.resize(image, [320, 320])
    return image


IMG_OUT_SIZE = 320


def get_boxes(item):
    x1 = tf.cast(item["X1"], tf.float32) * IMG_OUT_SIZE
    x2 = tf.cast(item["X2"], tf.float32) * IMG_OUT_SIZE
    y1 = tf.cast(item["Y1"], tf.float32) * IMG_OUT_SIZE
    y2 = tf.cast(item["Y2"], tf.float32) * IMG_OUT_SIZE

    w = x2 - x1
    h = y2 - y1

    # where each box is of the format [x, y, width, height]
    # box = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h], dtype=np.float32)

    gt_boxes = tf.cast(tf.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h]), tf.float32)
    gt_boxes = tf.reshape(gt_boxes, [-1, 4])

    return gt_boxes


# TODO: multiple objects in one image
# @tf.function
def process(item):
    labels_map = ["rv5", "rv8", "rv12"]

    image = load_image(item["PATH"])

    gt_boxes = get_boxes(item)

    # class_name = tf.cast(item["OBJECT"], tf.string)
    class_id = tf.cast(tf.where(labels_map == item["OBJECT"])[0], tf.float32)
    # gt_classes = tf.constant(class_id, dtype=tf.float32)

    gt_angles = tf.stack(
        [item["R11"], item["R21"], item["R31"], item["R12"], item["R22"], item["R32"]]
    )
    gt_angles = tf.cast(gt_angles, tf.float32)
    gt_angles = tf.reshape(gt_angles, [-1, 6])

    se = SamplesEncoder()

    label = se._encode_sample(image.shape, gt_boxes, class_id, gt_angles)

    return image, label


def create_dataset(meta_train, take_every=None):
    df = pd.read_csv(meta_train, header=None, names=column_names)
    df.head()

    if take_every is not None:
        hf = df.iloc[::take_every, :]
        dataset = tf.data.Dataset.from_tensor_slices(dict(hf))
    else:
        # random shuffle
        # see: https://stackoverflow.com/questions/42438591/shuffle-all-rows-of-a-csv-file-with-python
        hf = df.sample(frac=1)
        dataset = tf.data.Dataset.from_tensor_slices(dict(hf))

    return dataset.map(process)


# %%
def create_generator():
    def process(thumb):
        return thumb

    gen = BladeGenerator()
    # thumb, off_box, cMo = gen()

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec(shape=(320, 320, 3), dtype=tf.float32)),
    ).map(process)


if __name__ == "__main__":
    ds = create_generator()
    for e in ds:
        print(e)
# %%
