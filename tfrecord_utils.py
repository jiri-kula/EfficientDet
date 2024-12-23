#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import IPython.display as display
import cv2 as cv
import pandas as pd
from dataset_api import load_image, column_names
from progress.bar import Bar
import sys

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
    'drazka_rv5' : 0,
    'nedrazka_rv5' : 0,
    'drazka_rv8' : 1,
    'nedrazka_rv8' : 1,
    'drazka_rv12' : 2,
    'nedrazka_rv12' : 2,
    'roh' : 0,
    'vyrez' : 1,
}

path_map = {
  'X:\\Dataset\\zaznamy_z_vyroby\\' : "/home/jiri/remote_sd/DetectionData/Dataset/zaznamy_z_vyroby/",
  # '/home/jiri/winpart/' : '/home/jiri/remote_legion/winpart/',
  # '/home/jiri/DigitalAssistant/python/' : '/home/jiri/remote_legion/DigitalAssistant/python/',
  '/var/tmp/DetectionData/Dataset/' : '/home/jiri/remote_sd/DetectionData/Dataset/',
  'C:\\Edwards\\DetectionData\\Dataset\\' : '/home/jiri/winpart/Edwards/DetectionData/Dataset/',
  'C:/Edwards/' : '/home/jiri/winpart/Edwards/',
  '/mnt/c/Edwards/' : '/home/jiri/winpart/Edwards/',
  'C:/Users/nick/nullspaces/test-auto-annotation/testing/output/RV12/' : '/home/jiri/winpart/Edwards/annotation_mykyta/annotation/RV12/RV12/',
  'C:/Users/nick/nullspaces/test-auto-annotation/testing/output/dataset5-zdroj/drazka_rv5/' : '/home/jiri/winpart/Edwards/annotation_mykyta/annotation/RV5/RV5/RV5/dataset5-zdroj/drazka_rv5/',
  'C:/Users/nick/nullspaces/test-auto-annotation/testing/output/dataset5-zdroj/nedrazka_rv5/' : '/home/jiri/winpart/Edwards/annotation_mykyta/annotation/RV5/RV5/RV5/dataset5-zdroj/nedrazka_rv5/',
  'C:/Users/nick/nullspaces/test-auto-annotation/testing/output/hala/' : '/home/jiri/winpart/Edwards/annotation_mykyta/annotation/RV5/RV5/RV5/hala/',
  'C:/Users/nick/nullspaces/test-auto-annotation/testing/output/lcd/' : '/home/jiri/winpart/Edwards/annotation_mykyta/annotation/RV5/RV5/RV5/lcd/',
  'C:/Users/nick/nullspaces/test-auto-annotation/testing/output/ruka/' : '/home/jiri/winpart/Edwards/annotation_mykyta/annotation/RV5/RV5/RV5/ruka/',
  'C:/Users/nick/nullspaces/test-auto-annotation/testing/output/RV8/' : '/home/jiri/winpart/Edwards/annotation_mykyta/annotation/RV8/RV8/',
  
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
    object = rows.iloc[i]["OBJECT"]
    try:
      classes.append(image_labels[object])
    except:
      print(object, rows["PATH"].values)

    xmin = rows.iloc[i]["X1"]
    ymin = rows.iloc[i]["Y1"]

    xmax = rows.iloc[i]["X2"]
    ymax = rows.iloc[i]["Y2"]

    # check axis swap of x
    dx = xmax - xmin
    if dx == 0:
      raise ValueError("dx == 0 in ", rows["PATH"].values[0])

    if dx < 0:
      xmax, xmin = xmin, xmax

    # check axis swap of y
    dy = ymax - ymin
    if dy == 0:
      raise ValueError("dy == 0 in ", rows["PATH"].values[0])
    if dy < 0:
      ymax, ymin = ymin, ymax

    xmins.append(xmin)
    ymins.append(ymin)
    xmaxes.append(xmax)
    ymaxes.append(ymax)

    # check if rotation matrix is present in csv file
    if not np.isnan(rows.iloc[i]["R11"]):
      r11s.append(rows.iloc[i]["R11"])
      r21s.append(rows.iloc[i]["R21"])
      r31s.append(rows.iloc[i]["R31"])
      r12s.append(rows.iloc[i]["R12"])
      r22s.append(rows.iloc[i]["R22"])
      r32s.append(rows.iloc[i]["R32"])
    else:
      # if not, and object contains 'nedrazka', then set rotation matrix to identity
      # so that the 'x' axis is pointing to the right, 'z' axis is pointing up
      if rows.iloc[i]["OBJECT"].find("ne") > -1:
        r11s.append(0.0)
        r21s.append(0.0)
        r31s.append(1.0)
        r12s.append(0.0)
        r22s.append(-1.0)
        r32s.append(0.0)
      else:
        # if not, and object contains 'drazka', then set rotation matrix to identity
        # so that the 'x' axis is pointing to the left, 'z' axis is pointing up
        r11s.append(0.0)
        r21s.append(0.0)
        r31s.append(-1.0)
        r12s.append(0.0)
        r22s.append(-1.0)
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
        'image_raw': _bytes_feature(image_bytes.read()),
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
def process(meta_train):
  # set meta file path
  # meta_train = "/home/jiri/remote_seagate/LEGION5_DISK_D/DetectionData/Dataset/zaznamy_z_vyroby/2023_10_27-merge-all.csv"
  # meta_train = "/home/jiri/remote_sd/DetectionData/Dataset/zaznamy_z_vyroby/2023_10_27-merge-all.csv"
  # meta_train = "/home/jiri/DigitalAssistant/python/dataset6/images/meta.csv"
  tfrecord_path = '/home/jiri/tfrecords_allrot/' + os.path.split(meta_train)[0].replace("/", "_") + '.tfrecord'

  print("Processing: ", meta_train)
  print("Output: ", tfrecord_path)

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
  with tf.io.TFRecordWriter(tfrecord_path) as file_writer:
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
if __name__ == "__main__":
    process(sys.argv[1])
