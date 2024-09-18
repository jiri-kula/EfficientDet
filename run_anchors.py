# %%
from model.anchors import Anchors
import numpy as np

# %%
an = Anchors()
tensor = an.get_anchors(320, 320)

# Convert the tensor to a NumPy array
tensor_values = tensor.numpy()

# Open a text file in write mode
with open("anchors.txt", "w") as file:
    # Write the values to the text file
    np.savetxt(file, tensor_values, fmt="%s")

print("Tensor values have been written to tensor_values.txt")
# %%
from model.anchors import SamplesEncoder
import tensorflow_datasets as tfds

import tensorflow as tf

tf.config.run_functions_eagerly(True)


se = SamplesEncoder()


# @tf.autograph.experimental.do_not_convert
def decode_fn(sample):
    image = tf.cast(sample["image"], tf.float32)
    gt_classes = tf.cast(sample["objects"]["label"], tf.float32)

    gt_boxes = sample["objects"]["bbox"]
    gt_boxes = tf.multiply(gt_boxes, 320.0)
    y1, x1, y2, x2 = tf.split(gt_boxes, 4, axis=-1)
    gt_boxes = tf.concat([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1)

    def handle_empty_boxes():
        print("Empty gt_boxes detected, creating default label.")
        default_gt_boxes = tf.ones((1, 4), dtype=tf.float32)
        default_gt_classes = -1.0 * tf.ones((1,), dtype=tf.float32)
        default_gt_angles = tf.zeros((1, 6), dtype=tf.float32)
        return default_gt_boxes, default_gt_classes, default_gt_angles

    def handle_non_empty_boxes():
        batch_size = tf.shape(gt_boxes)[0]
        gt_angles = tf.zeros(shape=(batch_size, 6), dtype=tf.float32)
        return gt_boxes, gt_classes, gt_angles

    gt_boxes, gt_classes, gt_angles = tf.cond(
        tf.equal(tf.size(gt_boxes), 0), handle_empty_boxes, handle_non_empty_boxes
    )

    # batch_size = tf.shape(gt_boxes)[0]
    # gt_angles = tf.zeros(shape=(batch_size, 6), dtype=tf.float32)

    label = se._encode_sample(image.shape, gt_boxes, gt_classes, gt_angles)

    return image, label


train_data = tfds.load("kk_dataset", split="train", shuffle_files=False)
# .map(
#     decode_fn, num_parallel_calls=1  # tf.data.AUTOTUNE
# )

for item in train_data:
    if item["objects"]["bbox"].shape[0] != 0:
        continue
    decode_fn(item)
    print(item)
