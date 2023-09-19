# # %%
# import tensorflow as tf
# import pathlib
# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# from anchor_histogram import column_names
# from model.anchors import SamplesEncoder

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# np.set_printoptions(precision=4)

# meta_train = "/mnt/c/Edwards/rot_anot/RV12/drazka/drazka_rv12/meta.csv"
# ds = tf.data.experimental.make_csv_dataset(
#     meta_train,
#     batch_size=5,  # Artificially small to make examples easier to show.
#     num_epochs=1,
#     ignore_errors=True,
#     column_names=column_names,
# )

# for batch in ds.take(1):
#     for key, value in batch.items():
#         print("{:20s}: {}".format(key, value.numpy()))


# # %%
# # Reads an image from a file, decodes it into a dense tensor, and resizes it
# # to a fixed shape.
# @tf.function
# def make_images(batch):
#     for key, value in batch.items():
#         print("{:20s}: {}".format(key, value))

#     # purpose = features["PURPOSE"]
#     # path = features["PATH"]

#     # tf.print(path)

#     # label = []

#     # image = tf.io.read_file(path)
#     # image = tf.io.decode_jpeg(image)
#     # image = tf.image.convert_image_dtype(image, tf.float32)
#     # image = tf.image.resize(image, [128, 128])
#     # return image, label

#     return batch


# ds2 = ds.map(make_images)

# for batch in ds2.take(1):
#     for key, value in batch.items():
#         print("{:20s}: {}".format(key, value.numpy()))

# # %%
# df = pd.read_csv(meta_train, header=None, names=column_names)
# df.head()

# ds = tf.data.Dataset.from_tensor_slices(dict(df))

# # for feature_batch in ds.take(2):
# #     for key, value in feature_batch.items():
# #         print("  {!r:20s}: {}".format(key, value))

# IMG_OUT_SIZE = 320
# image_shape = (IMG_OUT_SIZE, IMG_OUT_SIZE)

# se = SamplesEncoder()


# def load_image(image_path):
#     Image = tf.keras.utils.load_img(image_path)
#     Image = Image.resize((IMG_OUT_SIZE, IMG_OUT_SIZE))
#     Image = tf.keras.utils.img_to_array(
#         Image, dtype=np.float32
#     )  # see inference.py / 255.0


# @tf.function
# def tform_sample(in_sample):
#     out_sample = in_sample

#     for key, value in in_sample.items():
#         print("  {!r:20s}: {}".format(key, value))
#     # out_sample = se._encode_sample(image_shape, gt_boxes, lbl_classes, lbl_angles)

#     return out_sample


# ds2 = ds.map(tform_sample)

# # for feature_batch in ds2.take(1):
# #     for key, value in feature_batch.items():
# #         print("  {!r:20s}: {}".format(key, value))

# # %%
# import tensorflow as tf

# from dataset import MyDataset, CSVDataset, image_mosaic, IMG_OUT_SIZE

# meta_train = "/mnt/c/Edwards/rot_anot/RV12/drazka/drazka_rv12/meta.csv"
# ds1 = CSVDataset(meta_train, aug=0, batch_size=4)

# BS = 5
# ds = tf.data.Dataset.from_generator(
#     CSVDataset,
#     args=[meta_train, 0, BS],
#     output_signature=(
#         tf.TensorSpec(shape=(BS, 320, 320, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(BS, 19206, 11), dtype=tf.float32),
#     ),
# )

# for batch in ds:
#     x, y = batch
# %%
import tensorflow as tf
import pandas as pd
import numpy as np

from model.anchors import SamplesEncoder

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

meta_train = "/mnt/c/Edwards/rot_anot/RV12/drazka/drazka_rv12/meta.csv"
df = pd.read_csv(meta_train, header=None, names=column_names)
df.head()

dataset = tf.data.Dataset.from_tensor_slices(dict(df))


def load_image(image_path):
    raw_image = tf.io.read_file(image_path)
    image = tf.image.decode_png(raw_image, channels=3)
    image = tf.image.resize(image, [320, 320])
    return image


IMG_OUT_SIZE = 320


def get_boxes(item):
    x1 = float(item["X1"]) * IMG_OUT_SIZE
    x2 = float(item["X2"]) * IMG_OUT_SIZE
    y1 = float(item["Y1"]) * IMG_OUT_SIZE
    y2 = float(item["Y2"]) * IMG_OUT_SIZE

    w = x2 - x1
    h = y2 - y1

    # where each box is of the format [x, y, width, height]
    # box = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h], dtype=np.float32)

    gt_boxes = tf.cast(tf.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h]), tf.float32)
    gt_boxes = tf.reshape(gt_boxes, [-1, 4])

    return gt_boxes


def process(item):
    image = load_image(item["PATH"])

    gt_boxes = get_boxes(item)

    gt_classes = tf.constant([4.0], dtype=tf.float32)

    gt_angles = tf.stack(
        [item["R11"], item["R21"], item["R31"], item["R12"], item["R22"], item["R32"]]
    )
    gt_angles = tf.cast(gt_angles, tf.float32)
    gt_angles = tf.reshape(gt_angles, [-1, 6])

    se = SamplesEncoder()

    label = se._encode_sample(image.shape, gt_boxes, gt_classes, gt_angles)

    return image, label


ds2 = dataset.map(process)

BATCH_SIZE = 32

train_data = ds2.shuffle(5000)
# train_data = train_data.padded_batch(BATCH_SIZE, padding_values=(0.0, 1e-8, -1.0))
train_data = train_data.padded_batch(BATCH_SIZE)
train_data = train_data.prefetch(tf.data.AUTOTUNE)
