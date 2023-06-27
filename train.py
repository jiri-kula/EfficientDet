"""Script for creating and training a new model."""

import tensorflow as tf
from model.efficientdet import get_efficientdet
from model.losses import EffDetLoss
from model.anchors import SamplesEncoder
from dataset import MyDataset

MODEL_NAME = "efficientdet_d0"

NUM_CLASSES = 80

EPOCHS = 300
BATCH_SIZE = 2

INITIAL_LR = 0.01
DECAY_STEPS = 433 * 155
init_lr = 0.001

LR = tf.keras.experimental.CosineDecay(init_lr, DECAY_STEPS, 1e-3)

DATA_PATH = "/home/jiri/keypoint_rcnn_training_pytorch/rv12_COCO_dataset/train"
CHECKPOINT_PATH = "/tmp/checkpoints/folder"

# TODO: LOAD YOUR TRAINING DATA
# TRAINING DATA SHOUD BE IN FORMAT (Image, Bounding boxes, Class labels)
# train_data = '/path/to/training/data'
train_data = MyDataset(DATA_PATH, None, BATCH_SIZE)

samples_encoder = SamplesEncoder()
autotune = tf.data.experimental.AUTOTUNE

# train_data = train_data.shuffle(5000)
# train_data = train_data.padded_batch(BATCH_SIZE, padding_values=(0.0, 1e-8, -1.0))
# train_data = train_data.map(samples_encoder.encode_batch, num_parallel_calls=autotune)
# train_data = train_data.prefetch(autotune)

model = get_efficientdet(MODEL_NAME, num_classes=NUM_CLASSES)
loss = EffDetLoss(num_classes=NUM_CLASSES)
opt = tf.keras.optimizers.SGD(LR, momentum=0.9)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH, save_weights_only=True
)


def norm(y_true, y_pred):
    loss = tf.norm(y_pred - y_true, axis=1)

    # reduction
    red = tf.reduce_mean(loss)

    # tf.print('y_true:', y_true)
    # tf.print('y_pred:', y_pred)
    # tf.print('d_norm: ', loss)
    # tf.print('reduce: ', red)
    return red


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=EffDetLoss(num_classes=NUM_CLASSES),
    # metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()],
)

# model.build((4, 256, 256, 3))
# model.summary()

# tf.keras.utils.plot_model(
#     model,
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir="TB",
# )

model.fit(train_data, epochs=EPOCHS, use_multiprocessing=False)
