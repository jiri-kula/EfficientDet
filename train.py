# %%
"""Script for creating and training a new model."""

import tensorflow as tf
import numpy as np
import keras
from model.efficientdet import get_efficientdet
from model.losses import EffDetLoss
from model.anchors import SamplesEncoder
from dataset import MyDataset, CSVDataset, image_mosaic


MODEL_NAME = "efficientdet_d0"

NUM_CLASSES = 6

EPOCHS = 40
BATCH_SIZE = 8

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

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=loss,
    run_eagerly=True,
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

# train_data = MyDataset(DATA_PATH, None, BATCH_SIZE)
train_data = CSVDataset("/home/jiri/EfficientDet/meta_split_test.csv", None, BATCH_SIZE)
test_data = CSVDataset("/home/jiri/EfficientDet/meta_split_test.csv", None, BATCH_SIZE)

# %%


class CustomCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        images, lbl = train_data.__getitem__(batch)
        # print(images.shape)
        image_mosaic(images)


history = model.fit(
    train_data,
    epochs=EPOCHS,
    workers=2,
    # use_multiprocessing=True,
    validation_data=test_data,  #   , callbacks=[CustomCallback()]
)

# # %%
# images, gt = train_data.__getitem__(0)

# model.evaluate(images, gt)
# p = model.predict(np.array(images[0]))

# # %% prediction
# pred = model(images)

# loss = EffDetLoss(num_classes=NUM_CLASSES)
# l = loss(labels, pred)

# print("loss: {:f}".format(l))
# # %%

# model(image) == model(image, training=True)

# # %%
# images, labels = train_data.__getitem__(0)
# boxes = labels[0]
# classes = labels[1]

# se = SamplesEncoder()
# se.encode_batch(images, boxes, classes)

# %%
