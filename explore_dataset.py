#%%
import tensorflow as tf
from tfrecord_decode import decode_raw
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from mpl_toolkits.axes_grid1 import ImageGrid
from drawing import show_gizmo

#%%
example_path = "/home/jiri/winpart/Edwards/tfrecords_allrot/_home_jiri_DigitalAssistant_python_dataset4_images.tfrecord"

# train_data = tf.data.TFRecordDataset(example_path).map(decode_fn)

im = []
boxes = []

# for images, labels in train_data.batch(4).take(1):
#     for image in images:
#         im.append(tf.cast(image, dtype=tf.uint8))
    
#     for label in labels:
#         indexes = tf.where(label[..., 10] >= 0.0)
#         bs = []
#         for idx in indexes:
#             anchor_encoding = label[idx, :4]

i = 0
for item in tf.data.TFRecordDataset([example_path]).map(decode_raw).skip(2).take(5):
    # image
    image = tf.io.decode_png(item["image_raw"])
    image = tf.image.resize(image, [320, 320])

    assert(image.shape[0] == image.shape[1] == 320)

    # classes
    gt_classes = tf.sparse.to_dense(item["classes"], default_value=-1.0)
    gt_classes = tf.reshape(gt_classes, [-1, 1])

    
    # boxes
    xmins = tf.sparse.to_dense(item["xmins"], default_value=0.)
    ymins = tf.sparse.to_dense(item["ymins"], default_value=0.)
    xmaxes = tf.sparse.to_dense(item["xmaxes"], default_value=0.)
    ymaxes = tf.sparse.to_dense(item["ymaxes"], default_value=0.)

    assert(tf.reduce_all(tf.greater(xmaxes, xmins)))
    assert(tf.reduce_all(tf.greater(ymaxes, ymins)))

    # angles
    r11 = tf.sparse.to_dense(item["r11s"], default_value=0.)
    r21 = tf.sparse.to_dense(item["r21s"], default_value=0.)
    r31 = tf.sparse.to_dense(item["r31s"], default_value=0.)
    r12 = tf.sparse.to_dense(item["r12s"], default_value=0.)
    r22 = tf.sparse.to_dense(item["r22s"], default_value=0.)
    r32 = tf.sparse.to_dense(item["r32s"], default_value=0.)

    if not (tf.reduce_all(tf.equal([r11, r21, r31, r12, r22, r32], 0.0))):
        tf.print(item["image_path"])
    tf.print("Rx:", [r11, r21, r31, r12, r22, r32])
    
    i += 1
    print('\r>> You have finished %d' % i, end="")
    sys.stdout.flush()
    # example = tf.train.Example()
    # example.ParseFromString(batch.numpy())
    # print(example)

    plt.imshow(tf.cast(image, tf.uint8))

    for i in range(0, len(xmins)):
        x1 = xmins[i] * 320
        y1 = ymins[i] * 320
        x2 = xmaxes[i] * 320
        y2 = ymaxes[i] * 320
        patch = plt.Rectangle(
            [x1, y1], x2-x1, y2-y1, fill=False, edgecolor=[0, 1, 0], linewidth=1
        )
        plt.gca().add_patch(patch)

        print([r11[i], r21[i], r31[i], r12[i], r22[i], r32[i]])
        show_gizmo([r11[i], r21[i], r31[i], r12[i], r22[i], r32[i]], plt.gca(), x1, y1, x2-x1, y2-y1)

    plt.show()

    
# fig = plt.figure(figsize=(4., 4.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(2, 2),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )

# for ax, image in zip(grid, im):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(image)
#     patch = plt.Rectangle(
#         [80, 100], 50, 90, fill=False, edgecolor=[0, 1, 0], linewidth=1
#     )
#     ax.add_patch(patch)


# plt.show()


# %%
