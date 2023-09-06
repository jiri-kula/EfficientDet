# %%
"""Script for creating and training a new model."""

import datetime
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

EPOCHS = 1000
EAGERLY = True
BATCH_SIZE = 4 if EAGERLY else 16

INITIAL_LR = 0.01
DECAY_STEPS = 433 * 155
init_lr = 0.001

LR = tf.keras.experimental.CosineDecay(init_lr, DECAY_STEPS, 1e-3)

# DATA_PATH = "/home/jiri/keypoint_rcnn_training_pytorch/rv12_COCO_dataset/train"
DATA_PATH = "/mnt/d/dev/keypoints/rv12_dataset_v2"

CHECKPOINT_PATH = "/tmp/checkpoints/folder"

model = get_efficientdet(MODEL_NAME, num_classes=NUM_CLASSES)
loss = EffDetLoss(num_classes=NUM_CLASSES)
opt = tf.keras.optimizers.SGD(LR, momentum=0.9)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH, save_weights_only=True
)


# https://stackoverflow.com/questions/55428731/how-to-debug-custom-metric-values-in-tf-keras
# @tf.function
def AngleMetric(y_true, y_pred):
    class_label = y_true[..., 10]

    angle_labels = y_true[..., 4:10]
    angle_preds = y_pred[..., 4:10]

    loss = angle_labels - angle_preds

    positive_mask = tf.cast(tf.greater(class_label, -1.0), tf.float32)

    r1_pred = angle_preds[..., :3]
    r1_true = angle_labels[..., :3]

    proj = tf.reduce_sum(
        tf.multiply(r1_pred, r1_true), -1
    )  # dot prod of true and pred vectors
    q = tf.where(positive_mask == 1.0, proj, 0.0)  # zero out non-relevat

    # we want all projections to be 1 (fit)

    normalizer = tf.reduce_sum(positive_mask, axis=-1)

    sample_metric = tf.math.divide_no_nan(tf.reduce_sum(q, axis=-1), normalizer)

    batch_metric = tf.reduce_mean(sample_metric)

    return batch_metric  # want to be 1


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    run_eagerly=EAGERLY,
    metrics=[AngleMetric],
)

# %%

# train_data = MyDataset(DATA_PATH, None, BATCH_SIZE)
meta_train = "/mnt/c/Edwards/rv5/Output/ruka_hala_6D_aug.csv"
meta_test = "/mnt/c/Edwards/rv5/Output/ruka_6D_aug_test.csv"

train_data = CSVDataset(meta_train, None, BATCH_SIZE)
test_data = CSVDataset(meta_test, None, BATCH_SIZE)

model.build(input_shape=(BATCH_SIZE, 320, 320, 3))
model.summary(show_trainable=True)

weights_dir = "tmp/fit_6D_aug_proj"
current_epoch = 37
model.load_weights(weights_dir + "/epoch_{:d}".format(current_epoch))


class CustomCallback(keras.callbacks.Callback):
    # def on_train_batch_end(self, batch, logs=None):
    #     images, lbl = train_data.__getitem__(batch)
    #     # print(images.shape)
    #     image_mosaic(images)

    def on_epoch_end(self, epoch, logs=None):
        global current_epoch
        current_epoch += 1
        model.save_weights(weights_dir + "/epoch_{:d}".format(current_epoch))


# %%
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# gen = shared_mem_multiprocessing(train_data, workers=4)


class Loss_reporter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        tf.summary.scalar(
            "class loss", data=self.model.loss.get_last_clf_loss(), step=epoch
        )
        # tf.summary.scalar("box loss", self.model.loss.last_box_loss, step=epoch)
        # tf.summary.scalar("angle loss", self.model.loss.last_ang_loss, step=epoch)


log_dir = "logs/fit/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S" + "/" + weights_dir
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
# file_writer.set_as_default()

history = model.fit(
    train_data,
    epochs=EPOCHS,
    workers=1,
    use_multiprocessing=False,
    validation_data=None,
    callbacks=[CustomCallback(), tensorboard_callback],
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
    for _ in range(100):
        data = np.random.rand(1, 320, 320, 3)
        yield [data.astype(np.float32)]


# Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model') # path to the SavedModel directory
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_dataset
# This ensures that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
# converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
# converter.experimental_new_converter = False
tflite_model = converter.convert()

# Save the model.
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
# %%

interpreter = tf.lite.Interpreter("model.tflite")
input = interpreter.get_input_details()[0]  # Model has single input.
output = interpreter.get_output_details()[0]
interpreter.allocate_tensors()  # Needed before execution!
input_data = tf.constant(50, shape=[1, 320, 320, 3], dtype="uint8")
interpreter.set_tensor(input["index"], input_data)
interpreter.invoke()
retval = interpreter.get_tensor(output["index"])
retval[0, ...]


# %%
model.predict(images[0])
# %%
