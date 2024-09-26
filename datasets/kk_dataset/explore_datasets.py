# %%
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds


# %%
ds = tfds.load("kk_dataset:1.0.1", split="train", shuffle_files=False)
print(ds)

# %% Iterate over a dataset
for example in ds.take(4):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    print(list(example.keys()))
    image = example["image"]
    objects = example["objects"]
    print(image.shape, objects)

# %% Benchmark your datasets
ds = ds.batch(32).prefetch(1)

tfds.benchmark(ds, batch_size=32)
tfds.benchmark(ds, batch_size=32)  # Second epoch much faster due to auto-caching

# %% Visualization
ds, info = tfds.load("kk_dataset", split="train", with_info=True)
tfds.as_dataframe(ds.take(8), info)
# %%
fig = tfds.show_examples(ds.take(64), info)

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %%
for sample in ds.take(1):
    image = example["image"]
    objects = example["objects"]

    fig, ax = plt.subplots(1)

    ax.imshow(tf.squeeze(image, axis=0))

    yxyx = tf.squeeze(objects["bbox"], axis=0)

    for row in range(5):  # yxyx.shape[0]):
        print(yxyx[row].numpy())
        ymin, xmin, ymax, xmax = yxyx[row].numpy() * 320
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin, str(row), color="white")

# %%
