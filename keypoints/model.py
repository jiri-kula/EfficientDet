# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
import keras

IMG_SIZE = 224
NUM_KEYPOINTS = 5 * 2
DATA_DIR = "/home/jiri/remote_x"  # sshfs X:/home/shared/ssd ~/remote_x

# %% Visualization
# tfds.as_dataframe(ds.take(8), info)
# fig = tfds.show_examples(ds.take(4), info)


# %%
def decode_fn(sample):
    image = tf.cast(sample["image"], tf.float32)

    gt_boxes = sample["objects"]["bbox"]
    # gt_boxes = tf.multiply(gt_boxes, 384.0)

    # def handle_empty_boxes():
    #     return tf.zeros(
    #         (5, 4), dtype=tf.float32
    #     )

    # def handle_non_empty_boxes():
    #     return gt_boxes

    # y1, x1, y2, x2 = tf.cond(
    #     tf.equal(tf.size(gt_boxes), 0), handle_empty_boxes, handle_non_empty_boxes
    # )

    y1, x1, y2, x2 = tf.split(gt_boxes, 4, axis=-1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Stack cx and cy along a new axis
    stacked = tf.stack([cx, cy], axis=1)

    # Reshape to interleave
    kps = tf.reshape(stacked, [1, 1, -1])

    return image, kps


# %%
ds = tfds.load(
    "kp_dataset:1.0.11",
    split="train",
    shuffle_files=False,
    with_info=False,
    data_dir=DATA_DIR,
)

ds = ds.map(decode_fn)

dv = tfds.load(
    "kp_dataset:1.0.0",
    split="train",
    shuffle_files=False,
    with_info=False,
    data_dir=DATA_DIR,
).map(decode_fn)


# %%
# Parts of this code come from here:
# https://github.com/benjiebob/StanfordExtra/blob/master/demo.ipynb
def visualize_keypoints(images, keypoints):
    fig, axes = plt.subplots(nrows=len(images), ncols=2, figsize=(12, 12))
    [ax.axis("off") for ax in np.ravel(axes)]

    for (ax_orig, ax_all), image, current_keypoint in zip(axes, images, keypoints):
        ax_orig.imshow(image)
        ax_all.imshow(image)

        current_keypoint = np.array(current_keypoint)
        x = current_keypoint[:, :, 0::2]
        y = current_keypoint[:, :, 1::2]
        # Since the last entry is the visibility flag, wue discard it.
        ax_all.scatter(
            [IMG_SIZE * x], [IMG_SIZE * y], c="white", marker="x", s=50, linewidths=1
        )

    plt.tight_layout(pad=1.0)
    plt.show()


images, keypoints = [], []

for image, keypoint in ds.take(2):
    images.append(image / 255.0)
    keypoints.append(keypoint)

visualize_keypoints(images, keypoints)


# %%
def get_model():
    # Load the pre-trained weights of MobileNetV2 and freeze the weights
    backbone = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    print(f"Shape after backbone: {x.shape}")

    x = layers.Dropout(0.5)(x)

    x = layers.SeparableConv2D(
        256,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="conv1",
    )(x)
    # x = layers.SeparableConv2D(
    #     16, kernel_size=5, strides=1, padding="same", activation="relu", name="conv2"
    # )(x)
    x = layers.SeparableConv2D(
        128,
        kernel_size=5,
        strides=1,
        padding="valid",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="conv3",
    )(x)

    x = layers.Dropout(0.5)(x)

    # x = layers.SeparableConv2D(
    #     64, kernel_size=5, strides=1, padding="same", activation="relu"
    # )(x)
    # print(f"Shape after SeparableConv2D 64: {x.shape}")

    # x = layers.SeparableConv2D(
    #     NUM_KEYPOINTS, kernel_size=3, strides=1, activation="relu"
    # )(x)
    # print(f"Shape after SeparableConv2D NUM_KEYPOINTS: {x.shape}")

    # Add GlobalAveragePooling2D to reduce spatial dimensions to 1x1
    # outputs = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.SeparableConv2D(
        NUM_KEYPOINTS,
        kernel_size=3,
        strides=1,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="conv4",
    )(x)

    print(f"Shape of outputs: {outputs.shape}")

    return keras.Model(inputs, outputs, name="keypoint_detector")


# %%
get_model().summary()
model = get_model()


def custom_mse_loss(y_true, y_pred):
    # Scale y_true and y_pred from normalized range (0, 1) to pixel range
    y_true_pixels = y_true * IMG_SIZE
    y_pred_pixels = y_pred * IMG_SIZE

    # Compute the Mean Squared Error in pixel space
    return tf.reduce_mean(tf.square(y_true_pixels - y_pred_pixels))


# %%
EPOCHS = 1000
BATCH_SIZE = 64

model.compile(loss=custom_mse_loss, optimizer=keras.optimizers.Adam(1e-4))
model.fit(ds.batch(BATCH_SIZE), epochs=EPOCHS, validation_data=dv.batch(BATCH_SIZE))


# %%
image, label = next(iter(ds.batch(2)))
prediction = model.predict(image)

visualize_keypoints(image / 255.0, prediction)
# %%
