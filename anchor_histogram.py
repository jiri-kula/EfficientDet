# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


meta_train = "/mnt/c/Edwards/annotation/RV12/merge-all.csv"

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

df = pd.read_csv(meta_train, header=None, names=column_names)

# %%
# drazky, které jsou špatně anotované jako osa x směrem ke kameře
x1 = df["X1"]
x2 = df["X2"]
y1 = df["Y1"]
y2 = df["Y2"]

ws = x2 - x1
hs = y2 - y1

aspects = ws / hs

a = np.array(aspects)
w = np.array(ws) * 320
h = np.array(hs) * 320

# %% K-means
kmeans = KMeans(init="random", n_clusters=5, n_init=10, max_iter=300, random_state=42)

# fit = kmeans.fit(np.vstack((ws, hs)).transpose())
fit = kmeans.fit(w.reshape(-1, 1))

areas = h * w
good_areas = np.array(
    sorted(np.sqrt(kmeans.fit(areas.reshape(-1, 1)).cluster_centers_))
).reshape(1, -1)
print("good_areas: ", good_areas)

# aspects = w / h
# %%
freq, edges = np.histogram(aspects)
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
