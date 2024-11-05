# %%
# https://www.youtube.com/watch?v=RPocW_aMZKE&t=1s&ab_channel=TensorFlow

from datasets.decode_function import build_dataset
from datasets.decode_function import decode_fn

import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# %%
ds = tfds.load("kk_dataset:1.2.3", split="train", shuffle_files=False)

# sample = next(iter(ds))

# decode_fn(sample)


# %%
# ds.skip(175)
bad_sample = next(iter(ds.skip(176)))
decode_fn(bad_sample)

# %%
data = ds.map(decode_fn)
image, label = next(iter(data))
plt.imshow(image.numpy() / 255)
# %%
for i, (image, label) in enumerate(data):
    print(i)
    # print(tf.where(label[:, 10] > -1.0))
    # print(i, image.shape, label.shape)
    # img = image.numpy()
    # plt.imshow(image / 255)
    # plt.show()
    # if i > 0:
    #     break

# %%
