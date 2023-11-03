#%%
import tensorflow as tf
from tfrecord_decode import decode_raw, decode_fn, raw2label
import os.path
from tfrecord_files import train_list

ds = tf.data.TFRecordDataset(train_list)

folders = []

for item in ds:
    record = decode_raw(next(iter(ds)))
    image_path = record["image_path"].numpy()
    folder = os.path.split(image_path)[0]

    if not folder in folders:
        folders.append(folder)
        print(folders)


print(folders)