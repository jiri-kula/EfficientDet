# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

TFLITE_CONVERSION = False

EAGERLY = False
tf.config.run_functions_eagerly(EAGERLY)
# if EAGERLY:
#     tf.data.experimental.enable_debug_mode()

import datetime, os
import numpy as np
import keras
from model.efficientdet import EfficientDet
from model.losses import EffDetLoss, AngleLoss
from model.anchors import SamplesEncoder, Anchors
import random

# from dataset import CSVDataset, image_mosaic, IMG_OUT_SIZE
from model.utils import to_corners

import dataset_api
from tfrecord_decode import decode_fn

EPOCHS = 200
BATCH_SIZE = 4 if EAGERLY else 16
checkpoint_dir = "checkpoints/kernel_13"

# laod list of tfrecord files
with open("list_12_norot_simple.txt") as file:
    train_list  = [line.rstrip() for line in file]
random.shuffle(train_list)

# print shuffeled tfrecord files
for item in train_list:
    if not os.path.isfile(item):
        raise ValueError(item)

train_data = tf.data.TFRecordDataset(
    # "/home/jiri/winpart/Edwards/tfrecords_allrot/_home_jiri_remote_sd_DetectionData_Dataset_zaznamy_z_vyroby_2023_03_08_rv12_09_47_27.tfrecord"
    train_list
).map(decode_fn)

# num_samples = train_data.cardinality().numpy()
train_data = train_data.shuffle(4096)
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

# gen_data = dataset_api.create_generator()

# %%
NUM_CLASSES = 3

model = EfficientDet(
    channels=64,
    num_classes=NUM_CLASSES,
    num_anchors=9,
    bifpn_depth=3,
    heads_depth=3,
    name="efficientdet_d0",
    export_tflite=TFLITE_CONVERSION,
)

# "efficientnet-lite0|resample_p6|fpn_cells|class_det|box_regressor"
model.var_freeze_expr = ("class_det|box_regressor")
print("var_freeze_expr: ", model.var_freeze_expr)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=None,
    run_eagerly=EAGERLY,
)

# %%
model.build(input_shape=(BATCH_SIZE, 320, 320, 3))
model.summary(show_trainable=True)

# checkpoints
completed_epochs = 0
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is None:
    print("Checkpoint not found.")
else:
    # the epoch the training was at when the training was last interrupted
    print("latest checkpoint: {:s}".format(latest))
    completed_epochs = int(latest.split("/")[-1].split("-")[1])
    model.load_weights(latest)
    print("Checkpoint found {}".format(latest))

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "weights-{epoch:02d}"),
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)


# %%

# tensorboard
# time based log_dir
log_dir = "logs/fit/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S" + "/" + checkpoint_dir
)

# constant log_dir
# log_dir = "logs/fit/" + checkpoint_dir
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    # log_dir=log_dir, histogram_freq=1, write_images=False, profile_batch="3,8"
    log_dir=log_dir,
    histogram_freq=1,
    write_images=False,
)

# train_data = ds2.shuffle(5000)
# train_data = train_data.padded_batch(BATCH_SIZE)
# train_data = train_data.prefetch(tf.data.AUTOTUNE)

if not TFLITE_CONVERSION:
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        workers=1,
        use_multiprocessing=False,
        validation_data=None,
        initial_epoch=completed_epochs,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            tf.keras.callbacks.TerminateOnNaN(),
        ],
    )

# %%
print("Conversion")
model.compute_output_shape((1, 320, 320, 3))

# https://www.tensorflow.org/lite/performance/post_training_quantization


def representative_dataset():
    data = train_data.take(10)
    for image, label in data:
        yield [image]


# def representative_dataset():
#     for _ in range(100):
#         data = 255.0 * np.random.rand(1, 320, 320, 3)
#         yield [data.astype(np.float32)]


# Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model') # path to the SavedModel directory
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_dataset
# This ensures that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
#     # tf.lite.OpsSet.SELECT_TF_OPS,
# ]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.experimental_new_converter = False
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.float32  # or tf.uint8
tflite_model = converter.convert()


# Save the model.
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
    print("Done writing model to drive.")

# %%
interpreter = tf.lite.Interpreter("model.tflite")
input = interpreter.get_input_details()[0]  # Model has single input.
output_boxes = interpreter.get_output_details()[0]  # Model has 3 outputs
output_angles = interpreter.get_output_details()[1]  # Model has 3 outputs
output_classes = interpreter.get_output_details()[2]  # Model has 3 outputs
interpreter.allocate_tensors()  # Needed before execution!

# constant input
# input_data = tf.constant(0, shape=[1, 320, 320, 3], dtype="uint8")
# interpreter.set_tensor(input["index"], input_data)

# input from dataset
# img = train_data[0][0]
# im = img[0]
# im8 = im.astype(np.uint8)
# im8e = tf.expand_dims(im8, axis=0)
# interpreter.set_tensor(input["index"], im8e)

exit()

# %%
# input loaded from image path
image_path = "/home/jiri/winpart/Edwards/annotation/RV12/robotic-3-aug//drazka_rv12/image_drazka_rv12_0011.png"
raw_image = tf.io.read_file(image_path)
image = tf.image.decode_image(raw_image, channels=3, dtype=tf.uint8)
image = tf.expand_dims(image, axis=0)
# image = tf.cast(image, tf.float32)
interpreter.set_tensor(input["index"], image)

interpreter.invoke()
boxes = interpreter.get_tensor(output_boxes["index"])
classes = interpreter.get_tensor(output_classes["index"])
retval = boxes
retval[0, ...]


# retval = retval.astype(np.float32)
# real = (retval - zero_point) * scale

ianchor = 6652
retval[0, ianchor, :]

# tf.sigmoid(real[0, ianchor, 10:])

# %%
model.predict(images[0])
# %%

se = SamplesEncoder()
se._anchors._compute_dims()
