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
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

image_labels = {
    'rv3' : 0,
    'rv5' : 0,
    'rv8' : 1,
    'rv12' : 2,
    'drazka_rv3' : 0,
    'nedrazka_rv3' : 0,
    'drazka_rv8' : 1,
    'nedrazka_rv8' : 1,
    'drazka_rv12' : 2,
    'nedrazka_rv12' : 2,
}

path_map = {
  'X:\\Dataset\\zaznamy_z_vyroby\\' : "/home/jiri/remote_seagate/LEGION5_DISK_D/DetectionData/Dataset/zaznamy_z_vyroby/",
  '/home/jiri/winpart/' : '/home/jiri/remote_legion/'
}

def image_example(rows):

  classes = []
  xmins = []
  ymins = []
  xmaxes = []
  ymaxes = []

  r11s = []
  r21s = []
  r31s = []

  r12s = []
  r22s = []
  r32s = []

  for i in range(len(rows)):
    classes.append(image_labels[rows.iloc[i]["OBJECT"]])
    xmins.append(rows.iloc[i]["X1"])
    ymins.append(rows.iloc[i]["Y1"])
    xmaxes.append(rows.iloc[i]["X2"])
    ymaxes.append(rows.iloc[i]["Y2"])

    if not rows.iloc[i]["X1"]:
      r11s.append(rows.iloc[i]["R11"])
      r21s.append(rows.iloc[i]["R21"])
      r31s.append(rows.iloc[i]["R31"])
      r12s.append(rows.iloc[i]["R12"])
      r22s.append(rows.iloc[i]["R22"])
      r32s.append(rows.iloc[i]["R32"])
    else:
      r11s.append(0.0)
      r21s.append(0.0)
      r31s.append(0.0)
      r12s.append(0.0)
      r22s.append(0.0)
      r32s.append(0.0)


  image_path = rows.iloc[0]["PATH"]
  
  # is there record to replace path?
  for item in path_map:
    idx = image_path.find(item)
    if idx != -1:
      parts = image_path.split(item, 2)
      image_path = path_map[item] + parts[1]
      image_path = image_path.replace("\\", "/")
      break

  with open(image_path, 'rb') as image_bytes:
    # image_string = tf.io.decode_png(image_bytes.read())

    feature = {
        'image_path': _bytes_feature(rows["PATH"].values[0].encode('utf-8')),
        'image_raw': _bytes_feature(image_bytes.read(1)),
        'classes' : _float_feature(classes),
        'xmins' : _float_feature(xmins),
        'ymins' : _float_feature(ymins),
        'xmaxes' : _float_feature(xmaxes),
        'ymaxes' : _float_feature(ymaxes),
        'r11s' : _float_feature(r11s),
        'r21s' : _float_feature(r21s),
        'r31s' : _float_feature(r31s),
        'r12s' : _float_feature(r12s),
        'r22s' : _float_feature(r22s),
        'r32s' : _float_feature(r32s)
    }

  return tf.train.Example(features=tf.train.Features(feature=feature))

#   #%%
# image_path = "/home/jiri/remote_seagate/LEGION5_DISK_D/DetectionData/Dataset/RV12/hala_train/drazka/image_drazka_0019.png"

# with open(image_path, 'rb') as image_bytes:
#   image_example(image_bytes.read(), image_labels['rv12'])
# %%

# set meta file path
# meta_train = "/home/jiri/remote_seagate/LEGION5_DISK_D/DetectionData/Dataset/zaznamy_z_vyroby/2023_10_27-merge-all.csv"
meta_train = "/home/jiri/remote_legion/winpart/Edwards/DetectionData/Dataset/zaznamy_z_vyroby/2023_10_27-merge-all.csv"
# meta_train = "/home/jiri/remote_legion/winpart/Edwards/annotation/RV12/merge-e.csv"

# load csv as pandas DataFrame
df = pd.read_csv(meta_train, header=None, names=column_names)
df = df.sort_values(by=["PATH"]) # removing sort breaks the algorithm

bar = Bar(
    "Build cache",
    max=len(df),
    suffix="%(percent).1f%% - %(eta)ds",
)

N = len(df)
row = 0
with tf.io.TFRecordWriter('zaznamy_z_vyroby.tfrecord') as file_writer:
  while row < N:
    # take image path from the first row
    the_path = df.iloc[row].PATH

    last_row = row + 1
    while last_row < N and df.iloc[last_row]["PATH"] == the_path:
      last_row += 1

    # create tf.Example (tfrecotrd)
    sample = image_example(df[row:last_row])

    # write to tfrecord file
    file_writer.write(sample.SerializeToString())
      
    num_samples = last_row - row
    row = last_row

    bar.next(num_samples)
  bar.finish()
# %%
