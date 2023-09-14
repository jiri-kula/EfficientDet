# %%
"""Script for creating and training a new model."""

import datetime, os
import tensorflow as tf
import numpy as np
import keras
from model.efficientdet import get_efficientdet
from model.losses import EffDetLoss, AngleLoss
from model.anchors import SamplesEncoder, Anchors
from dataset import MyDataset, CSVDataset, image_mosaic, IMG_OUT_SIZE
from model.utils import to_corners

# tf.config.run_functions_eagerly(True)

MODEL_NAME = "efficientdet_d0"

NUM_CLASSES = 6

EPOCHS = 300
EAGERLY = False
BATCH_SIZE = 4 if EAGERLY else 32

INITIAL_LR = 0.01
DECAY_STEPS = 433 * 155
init_lr = 0.001

LR = tf.keras.experimental.CosineDecay(init_lr, DECAY_STEPS, 1e-3)

model = get_efficientdet(MODEL_NAME, num_classes=NUM_CLASSES)

model.var_freeze_expr = "efficientnet-lite0|resample_p6"

loss = EffDetLoss(num_classes=NUM_CLASSES)
opt = tf.keras.optimizers.SGD(LR, momentum=0.9)


# model.metrics.append(AngleMetric)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    run_eagerly=EAGERLY,
    # metrics=[AngleMetric],
)

# %%

# train_data = MyDataset(DATA_PATH, None, BATCH_SIZE)
meta_train = "/mnt/c/Edwards/rv5/Output/hala_ruka_6D_aug.csv"
# meta_test = "/mnt/c/Edwards/rv5/Output/ruka_6D_aug_test.csv"

train_data = CSVDataset(meta_train, None, BATCH_SIZE)
# test_data = CSVDataset(meta_test, None, BATCH_SIZE)

model.build(input_shape=(BATCH_SIZE, 320, 320, 3))
model.summary(show_trainable=True)

# checkpoints
checkpoint_dir = "checkpoints/hala_ruka_6D_aug_sae_norm"
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
    log_dir=log_dir, histogram_freq=1, write_images=True
)


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
class PostProcessedEfficientDet(tf.keras.Model):
    def __init__(self, efficient_det):
        super().__init__(name="PostProcessedEfficientDet")

        # self.box_variance = tf.cast([0.1, 0.1, 0.2, 0.2], tf.float32)
        # an = Anchors()
        # self.anchor_boxes = an.get_anchors(IMG_OUT_SIZE, IMG_OUT_SIZE)
        self.efficient_det = efficient_det

    def call(self, inputs):
        preds = self.efficient_det(inputs)

        boxes = preds[..., :4]
        # boxes = preds[..., :4] * self.box_variance

        # boxes = tf.concat(
        #     [
        #         boxes[..., :2] * self.anchor_boxes[..., 2:] + self.anchor_boxes[..., :2],
        #         tf.exp(boxes[..., 2:]) * self.anchor_boxes[..., 2:],
        #     ],
        #     axis=-1,
        # )

        # boxes = to_corners(boxes)
        angles = preds[..., 4:6]
        classes = tf.nn.sigmoid(preds[..., 6:])

        return boxes, classes, angles

        # nms = tf.image.combined_non_max_suppression(
        #     tf.expand_dims(boxes, axis=2),
        #     classes,
        #     max_output_size_per_class=4,
        #     max_total_size=8,
        #     iou_threshold=0.5,
        #     score_threshold=float('-inf'),
        #     clip_boxes=False,
        # )

        # return nms.valid_detections, nms.nmsed_boxes, nms.nmsed_scores, nms.nmsed_classes


post_model = PostProcessedEfficientDet(model)

post_model.compute_output_shape((1, 320, 320, 3))


# %%
model.compute_output_shape((1, 320, 320, 3))


def representative_dataset():
    for _ in range(2):
        data = train_data[_][0]
        yield [data]


# Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model') # path to the SavedModel directory
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_dataset
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
# converter.target_spec.supported_types = [tf.int8]
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
image_path = "/mnt/c/Edwards/rv5/Output/hala_6D/drazka_rv5/image_drazka_rv5_0000.png"
raw_image = tf.io.read_file(image_path)
image = tf.image.decode_image(raw_image, channels=3)
image = tf.expand_dims(image, axis=0)
interpreter.set_tensor(input["index"], image)

interpreter.invoke()
retval = interpreter.get_tensor(output["index"])
retval[0, ...]

# dequantize
scales = output["quantization_parameters"]["scales"]
scale = scales[0] if len(scales) > 0 else 1.0

zero_points = output["quantization_parameters"]["zero_points"]
zero_point = zero_points[0] if len(zero_points) > 0 else 0

real = (retval.astype(np.float32) - zero_point) * scale

# retval[0, 7376, 10:]
tf.sigmoid(real[0, 7376, 10:])

# %%
model.predict(images[0])
# %%
