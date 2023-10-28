#%%
import tensorflow as tf
from tfrecord_decode import decode_raw
import matplotlib.pyplot as plt
import numpy as np
import sys

from mpl_toolkits.axes_grid1 import ImageGrid

#%%
example_path = "zaznamy_z_vyroby.tfrecord"
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
for item in tf.data.TFRecordDataset([example_path]).map(decode_raw):
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

    i += 1
    print('\r>> You have finished %d' % i, end="")
    sys.stdout.flush()
    # example = tf.train.Example()
    # example.ParseFromString(batch.numpy())
    # print(example)

    
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, image in zip(grid, im):
    # Iterating over the grid returns the Axes.
    ax.imshow(image)
    patch = plt.Rectangle(
        [80, 100], 50, 90, fill=False, edgecolor=[0, 1, 0], linewidth=1
    )
    ax.add_patch(patch)


plt.show()

