# %%
"""Script for creating and training a new model."""

import datetime, os
import tensorflow as tf
import numpy as np
import keras
from model.efficientdet import get_efficientdet, AngleMetric
from model.losses import EffDetLoss, AngleLoss
from model.anchors import SamplesEncoder, Anchors
from dataset import CSVDataset, image_mosaic, IMG_OUT_SIZE
from model.utils import to_corners

from dataset_api import ds2

MODEL_NAME = "efficientdet_d0"

NUM_CLASSES = 6

EPOCHS = 300
EAGERLY = False
tf.config.run_functions_eagerly(EAGERLY)
BATCH_SIZE = 1 if EAGERLY else 64


model = get_efficientdet(MODEL_NAME, num_classes=NUM_CLASSES)
model.var_freeze_expr = "efficientnet-lite0|resample_p6"

loss = EffDetLoss(num_classes=NUM_CLASSES)

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)
optimizer = tf.keras.optimizers.SGD()  # (learning_rate=0.001, momentum=0.9)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # optimizer=optimizer,
    loss=loss,
    run_eagerly=EAGERLY,
    # metrics = [] see train_step of efficientdet
)

# %%

# train_data = MyDataset(DATA_PATH, None, BATCH_SIZE)
meta_train = "/mnt/c/Edwards/rot_anot/RV12/drazka/drazka_rv12/meta.csv"
# meta_test = "/mnt/c/Edwards/rv5/Output/ruka_6D_aug_test.csv"

# train_data = CSVDataset(meta_train, None, BATCH_SIZE)

# train_data = tf.data.Dataset.from_generator(
#     CSVDataset,
#     args=[meta_train, 0, BATCH_SIZE],
#     output_signature=(
#         tf.TensorSpec(shape=(BATCH_SIZE, 320, 320, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(BATCH_SIZE, 19206, 11), dtype=tf.float32),
#     ),
# ).prefetch(tf.data.AUTOTUNE)

# test_data = CSVDataset(meta_test, None, BATCH_SIZE)
# print("cardinality")
# tf.data.experimental.cardinality(train_data)

model.build(input_shape=(BATCH_SIZE, 320, 320, 3))
model.summary(show_trainable=True)

# checkpoints
checkpoint_dir = "checkpoints/rv12_adam"
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

train_data = ds2.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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


history = model.fit(
    train_data,
    epochs=EPOCHS,
    workers=1,
    use_multiprocessing=False,
    validation_data=None,
    initial_epoch=completed_epochs,
    callbacks=[model_checkpoint_callback, tensorboard_callback],
)

# %%
model.compute_output_shape((1, 320, 320, 3))

# https://www.tensorflow.org/lite/performance/post_training_quantization


def representative_dataset():
    data = train_data.take(1)
    for image, label in data:
        yield [image]


# def representative_dataset():
#     for _ in range(100):
#         data = 2.0 * np.random.rand(1, 320, 320, 3) - 1.0
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
# converter.target_spec.supported_ops = [
# tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
#     tf.lite.OpsSet.TFLITE_BUILTINS,
# tf.lite.OpsSet.SELECT_TF_OPS,
# ]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_model = converter.convert()


# Save the model.
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
    print("Done writing model to drive.")

# %%
interpreter = tf.lite.Interpreter("model.tflite")
input = interpreter.get_input_details()[0]  # Model has single input.
output = interpreter.get_output_details()[0]
interpreter.allocate_tensors()  # Needed before execution!¨

# constant input
# input_data = tf.constant(0, shape=[1, 320, 320, 3], dtype="uint8")
# interpreter.set_tensor(input["index"], input_data)

# input from dataset
# img = train_data[0][0]
# im = img[0]
# im8 = im.astype(np.uint8)
# im8e = tf.expand_dims(im8, axis=0)
# interpreter.set_tensor(input["index"], im8e)

# input loaded from image path
image_path = (
    "/mnt/c/Edwards/rot_anot2/RV12/drazka/drazka_rv12/image_drazka_rv12_0000.png"
)
raw_image = tf.io.read_file(image_path)
image = tf.image.decode_image(raw_image, channels=3, dtype=tf.uint8)
image = tf.expand_dims(image, axis=0)
interpreter.set_tensor(input["index"], image)

interpreter.invoke()
retval = interpreter.get_tensor(output["index"])
retval[0, ...]

# dequantize
scales = output["quantization_parameters"]["scales"]
scale = scales[0] if len(scales) > 0 else 1.0

zero_points = output["quantization_parameters"]["zero_points"]
zero_point = zero_points[0].astype(np.float32) if len(zero_points) > 0 else 0.0

retval = retval.astype(np.float32)
real = (retval - zero_point) * scale

ianchor = 6652
retval[0, ianchor, :]

# tf.sigmoid(real[0, ianchor, 10:])

# %%
model.predict(images[0])
# %%
