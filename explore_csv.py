#%%
import pandas as pd

from dataset_api import column_names

meta_train = "/home/jiri/winpart/Edwards/annotation/RV12/merge-e.csv"

df = pd.read_csv(meta_train, header=None, names=column_names)

paths = df["PATH"]

#%%
df.loc[paths.str.contains("dataset6")]
# pd.unique(df["PATH"])