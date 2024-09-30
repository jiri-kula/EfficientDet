import tensorflow as tf
from model.anchors import SamplesEncoder
import imgaug.augmenters as iaa
import numpy as np
import tensorflow_datasets as tfds

se = SamplesEncoder()

# Define an augmentation sequence
augmentation_seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # add Gaussian noise
        iaa.Multiply((0.8, 1.2)),  # change brightness
        iaa.LinearContrast((0.75, 1.5)),  # change contrast
        iaa.GaussianBlur(sigma=(0.0, 3.0)),  # apply Gaussian blur
        iaa.SaltAndPepper(0.05),  # add salt and pepper noise
        iaa.CoarseDropout(0.02, size_percent=0.5),  # randomly drop rectangular regions
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5)),  # sharpen the image
        iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),  # emboss the image
        iaa.AdditivePoissonNoise(lam=(0.0, 10.0)),  # add Poisson noise
        iaa.Dropout(p=(0.01, 0.1)),  # randomly drop pixels
        iaa.Invert(0.05, per_channel=True),  # invert colors
        iaa.Solarize(0.1, threshold=(32, 128)),  # solarize the image
        iaa.Posterize(),  # reduce the number of bits for each color channel
    ]
)


def augment_image(image):
    # Randomly flip the image horizontally
    # image = tf.image.random_flip_left_right(image)

    # Randomly flip the image vertically
    # image = tf.image.random_flip_up_down(image)

    # Randomly adjust brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Randomly adjust contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Randomly adjust saturation
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

    # Randomly adjust hue
    # image = tf.image.random_hue(image, max_delta=0.1)

    return image


# Function to apply augmentation using imgaug
def apply_augmentation(image):
    image_np = image.numpy().astype(np.uint8)
    image_aug = augmentation_seq(image=image_np)
    return image_aug


# @tf.autograph.experimental.do_not_convert
def decode_fn(sample):
    image = tf.cast(sample["image"], tf.float32)
    gt_classes = tf.cast(sample["objects"]["label"], tf.float32)

    # # Use tf.py_function to apply augmentation
    # image_aug = tf.py_function(func=apply_augmentation, inp=[image], Tout=tf.float32)
    # image_aug.set_shape(image.shape)  # Set the shape of the augmented image
    image_aug = augment_image(image)

    image_aug = tf.image.resize(image_aug, (384, 384))

    gt_boxes = sample["objects"]["bbox"]
    gt_boxes = tf.multiply(gt_boxes, 384.0)
    y1, x1, y2, x2 = tf.split(gt_boxes, 4, axis=-1)
    gt_boxes = tf.concat([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1)

    def handle_empty_boxes():
        print("Empty gt_boxes detected, creating default label.")
        default_gt_boxes = tf.ones(
            (1, 4), dtype=tf.float32
        )  # using ones to avoid log(0) in box_target
        default_gt_classes = tf.zeros((1,), dtype=tf.float32)
        default_gt_angles = tf.zeros((1, 6), dtype=tf.float32)
        return default_gt_boxes, default_gt_classes, default_gt_angles

    def handle_non_empty_boxes():
        batch_size = tf.shape(gt_boxes)[0]
        gt_angles = tf.zeros(shape=(batch_size, 6), dtype=tf.float32)
        return gt_boxes, gt_classes, gt_angles

    gt_boxes, gt_classes, gt_angles = tf.cond(
        tf.equal(tf.size(gt_boxes), 0), handle_empty_boxes, handle_non_empty_boxes
    )

    # batch_size = tf.shape(gt_boxes)[0]
    # gt_angles = tf.zeros(shape=(batch_size, 6), dtype=tf.float32)

    label = se._encode_sample(image_aug.shape, gt_boxes, gt_classes, gt_angles)

    return image_aug, label


# Function to concatenate multiple datasets
def concat_datasets():
    ds0 = tfds.load("kk_dataset:1.0.0", split="train", shuffle_files=True)
    ds1 = tfds.load("kk_dataset:1.0.1", split="train", shuffle_files=True)
    ds2 = tfds.load("kk_dataset:1.0.2", split="train", shuffle_files=True)
    ds3 = tfds.load("kk_dataset:1.0.3", split="train", shuffle_files=True)
    ds4 = tfds.load("kk_dataset:1.0.4", split="train", shuffle_files=True)

    return ds0.concatenate(ds1).concatenate(ds2).concatenate(ds3).concatenate(ds4)


# Function to create trainig samples for the model trainig
def build_dataset(batch_size):
    train_data = concat_datasets().map(decode_fn, num_parallel_calls=1)

    train_data = train_data.shuffle(100)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    return train_data
