# %%

# Train checklist
# [] Create tfrecord out of csv file
# [] Set model/anchors.py areas so that first smallest leve is mean of histogram values given by anchor_histogram_tfrecord.py
# [] Set checkpoint_dir
# [] Set var_freeze_expr

# SDG
# lr = (0.0001-0.0005), momentum=0.9

# Notes
# Training accuracy higly dependend on proper initialization of anchors

import os
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from model.efficientdet import EfficientDet
from model.anchors import SamplesEncoder
from model.utils import to_corners
from dataset_api import create_dataset
from datasets.decode_function import decode_fn

EAGERLY = False
NUM_CLASSES = 2
BATCH_SIZE = 4 if EAGERLY else 16
EPOCHS = 50
TFLITE_CONVERSION = False
VAR_FREEZE_EXPR = "efficientnet-lite0"  # "efficientnet-lite0|resample_p6|fpn_cells"

checkpoint_dir = "checkpoints/v1_0_3_4_var_none"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.run_functions_eagerly(EAGERLY)

ds0 = tfds.load("kk_dataset:1.0.0", split="train", shuffle_files=True)
ds1 = tfds.load("kk_dataset:1.0.1", split="train", shuffle_files=True)
ds2 = tfds.load("kk_dataset:1.0.2", split="train", shuffle_files=True)
ds3 = tfds.load("kk_dataset:1.0.3", split="train", shuffle_files=True)
ds4 = tfds.load("kk_dataset:1.0.4", split="train", shuffle_files=True)

train_data = ds3.concatenate(ds4).map(
    decode_fn, num_parallel_calls=1
)  # tf.data.AUTOTUNE
train_data = train_data.shuffle(1000)
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

# %%
model = EfficientDet(
    channels=64,
    num_classes=NUM_CLASSES,
    num_anchors=3,
    bifpn_depth=3,
    heads_depth=3,
    name="efficientdet_d0",
    export_tflite=TFLITE_CONVERSION,
)

model.var_freeze_expr = VAR_FREEZE_EXPR  # "efficientnet-lite0|resample_p6|fpn_cells"
print("var_freeze_expr: ", model.var_freeze_expr)

print("model compilation", end=" ")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss=None,
    run_eagerly=EAGERLY,
)
print(" done")

# %%
print("model build", end=" ")
model.build(input_shape=(BATCH_SIZE, 320, 320, 3))
print(" done")

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

# tensorboard
# time based log_dir
log_dir = "logs/fit/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S" + "/" + checkpoint_dir
)

print("log_dir: ", log_dir)

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
if not TFLITE_CONVERSION:
    print("Training done.")
    exit()

print("Conversion")
model.compute_output_shape((1, 320, 320, 3))

# https://www.tensorflow.org/lite/performance/post_training_quantization


def representative_dataset_gen():
    for input_value in train_data.take(1):
        image, label = input_value
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
converter.representative_dataset = representative_dataset_gen
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

exit()
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

# exit()

# %%
# input loaded from image path
image_path = "/home/jiri/kk_csv/image_000001.png"
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

# %%
