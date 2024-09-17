# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# from tfrecord_files import train_list
from tfrecord_decode import decode_raw, raw2label

# laod list of tfrecord files
with open("list.txt") as file:
    train_list = [line.rstrip() for line in file]

ds = tf.data.TFRecordDataset(train_list).map(decode_raw)

# %%
ws = []
hs = []

for item in ds:
    image, gt_boxes, gt_classes, gt_angles = raw2label(item)
    for box in gt_boxes:
        ws.append(box[2].numpy())
        hs.append(box[3].numpy())

w = np.array(ws)
h = np.array(hs)

a = w / h


def stats(x):
    print("min: {:f}\nmax: {:f}".format(min(x), max(x)))


stats(w)
stats(a)

# here w and h are widths and heights of bounding boxes found in the dataset, need to compute aspect rations, scales and areas to setup anchors.py


# %% K-means
kmeans = KMeans(init="random", n_clusters=5, n_init=10, max_iter=300, random_state=42)

# fit = kmeans.fit(np.vstack((ws, hs)).transpose())
fit = kmeans.fit(w.reshape(-1, 1))

areas = h * w
good_areas = np.array(
    sorted(np.sqrt(kmeans.fit(areas.reshape(-1, 1)).cluster_centers_))
).reshape(1, -1)
print("good_areas: ", np.round(good_areas))


# aspects = w / h
# %%
freq, edges = np.histogram(a)
center = (edges[1:] + edges[:-1]) / 2
plt.bar(center, freq)


# %%
def stats(x):
    print("min: {:f}\nmax: {:f}".format(min(x), max(x)))


print("width\n---------------------------")
stats(w)

print("height\n---------------------------")
stats(h)

# %% Aspect ratios
kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)

fit = kmeans.fit(a.reshape(-1, 1))

aspects = np.array(sorted(fit.cluster_centers_)).reshape((1, -1))
print("aspects: ", aspects)

# %%


def split(areas, N):
    area_min = min(areas)
    area_max = max(areas)
    h = (area_max - area_min) / N
    area_center = np.linspace(area_min + h / 2, area_max - h / 2, N)

    return area_center


print("areas:", np.round(np.sqrt(split(w * h, 5))))
print("aspects:", split(w / h, 3))
# %%
# %%
