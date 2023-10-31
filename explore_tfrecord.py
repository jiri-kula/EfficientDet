#%%
import tensorflow as tf
from tfrecord_decode import decode_raw, decode_fn, raw2label
import os.path

ds = tf.data.TFRecordDataset(
    [
        "/home/jiri/winpart/Edwards/merge-e.tfrecord",  # this is mykyta + dataset4 + dataset6
        "/home/jiri/winpart/Edwards/zaznamy_z_vyroby.tfrecord",  # this is only zaznamy_z_vyroby
    ]
)

folders = []

for item in ds:
    record = decode_raw(next(iter(ds)))
    image_path = record["image_path"].numpy()
    folder = os.path.split(image_path)[0]

    if not folder in folders:
        folders.append(folder)
        print(folders)


print(folders)