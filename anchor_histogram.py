# %%
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

meta_train = "/mnt/c/Edwards/rot_anot/RV12/drazka/drazka_rv12/meta.csv"

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
w = np.array(ws)
h = np.array(hs)

# %% K-means
kmeans = KMeans(init="random", n_clusters=5, n_init=10, max_iter=300, random_state=42)

# fit = kmeans.fit(np.vstack((ws, hs)).transpose())
fit = kmeans.fit(w.reshape(-1, 1))

areas = 320**2 * h * w
sorted(np.sqrt(kmeans.fit(areas.reshape(-1, 1)).cluster_centers_))

# aspects = w / h
# %%
