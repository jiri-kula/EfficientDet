#%%
import tensorflow as tf

import numpy as np
import IPython.display as display
import cv2 as cv
import pandas as pd
from dataset_api import load_image, column_names
from progress.bar import Bar

# %%
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

image_labels = {
    'rv3' : 0,
    'rv8' : 1,
    'rv12' : 2,
}

def image_example(image_string, label):
  image_shape = tf.io.decode_png(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
      'bboxes' : _float_feature(xmins)
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

#   #%%
# image_path = "/home/jiri/remote_seagate/LEGION5_DISK_D/DetectionData/Dataset/RV12/hala_train/drazka/image_drazka_0019.png"

# with open(image_path, 'rb') as image_bytes:
#   image_example(image_bytes.read(), image_labels['rv12'])
# %%

# set meta file path
meta_train = "/home/jiri/remote_seagate/LEGION5_DISK_D/DetectionData/Dataset/zaznamy_z_vyroby/2022_12_20/rv12/montaz_1/meta.csv"
# meta_train = "/home/jiri/remote_legion/Edwards/annotation/RV12/merge-e.csv"

# load csv as pandas DataFrame
df = pd.read_csv(meta_train, header=None, names=column_names)
df = df.sort_values(by=["PATH"])

bar = Bar(
    "Build cache",
    max=len(df),
    suffix="%(percent).1f%% - %(eta)ds",
)

SEARCH_LEN = 1000
N = len(df)
row = 0
while row < N:
  # take image path from the first row
  the_path = df.iloc[row].PATH

  # get indexes of the same image path
  search_len = min(SEARCH_LEN, N-row)
  # subset = df[row:row + search_len] # expecting no more that 10 blades in one image
  # same_paths_idx = row + subset.index[subset['PATH'] == the_path]
  same_paths_idx = df.index[df['PATH'] == the_path]

  # create tf.Example (tfrecotrd)

  # remove these rows from the DataFrame
  # df.drop(same_paths_idx, inplace=True)
  row += len(same_paths_idx)

  bar.next()
# %%
