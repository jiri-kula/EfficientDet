"""Inference script."""

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model.efficientdet import get_efficientdet
from model.anchors import Anchors
from model.utils import to_corners, resize_and_pad

IMG_SIZE = 320


parser = argparse.ArgumentParser(description="Detect objects on image.")
parser.add_argument(
    "-n",
    metavar="NAME",
    default="efficientdet_d0",
    required=True,
    help="Name of model to use",
)
parser.add_argument(
    "-w", metavar="WEIGHTS", required=True, help="Path to model weights"
)
parser.add_argument("-i", metavar="IMAGE", required=True, help="Image to process")
parser.add_argument(
    "-c",
    metavar="CLASSES",
    default=80,
    type=int,
    help="Number of classes pretrained model predicts.",
)
parser.add_argument(
    "-a",
    metavar="ANCHORS",
    default=9,
    type=int,
    help="Number of anchor boxes pretrained model predicts.",
)
parser.add_argument(
    "-o", metavar="OUTPUT_NAME", default="output.png", help="Name of result image"
)


def make_prediction(
    image,
    max_output_size_per_class=100,
    max_total_size=100,
    iot_threshold=0.7,
    score_threshold=0.7,
):
    box_variance = tf.cast([0.1, 0.1, 0.2, 0.2], tf.float32)

    padded_image, new_shape, scale = resize_and_pad(
        image, target_side=float(IMG_SIZE), scale_jitter=None, stride=320.0
    )
    an = Anchors()
    anchor_boxes = an.get_anchors(padded_image.shape[0], padded_image.shape[1])

    preds = model.predict(tf.expand_dims(padded_image, axis=0))

    boxes = preds[..., :4] * box_variance
    boxes = tf.concat(
        [
            boxes[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
            tf.exp(boxes[..., 2:]) * anchor_boxes[..., 2:],
        ],
        axis=-1,
    )
    boxes = to_corners(boxes)
    angles = preds[..., 4:10]
    classes = tf.nn.sigmoid(preds[..., 10:])

    valid_dets = 0
    while valid_dets < 1 and score_threshold > 0:
        nms = tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            classes,
            max_output_size_per_class=max_output_size_per_class,
            max_total_size=max_total_size,
            iou_threshold=iot_threshold,
            score_threshold=score_threshold,
            clip_boxes=False,
        )

        valid_dets = nms.valid_detections[0]

        score_threshold -= 0.1

    max_anchor_scores = tf.reduce_max(classes, axis=-1)

    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()

    for i in range(valid_dets):
        x_min, y_min, x_max, y_max = nms.nmsed_boxes[0, i] / scale
        w, h = x_max - x_min, y_max - y_min
        # x_min, y_min, w, h = 75, 40, 35, 20
        patch = plt.Rectangle(
            [x_min, y_min], w, h, fill=False, edgecolor=[0, 1, 0], linewidth=1
        )
        ax.add_patch(patch)

        angle_idx = tf.where(max_anchor_scores[0] == nms.nmsed_scores[0, i])
        angle = angles[0, int(angle_idx)]

        oz = angle
        oz /= np.linalg.norm(oz)

        oy = np.array([-oz[1], oz[0]])

        cx = x_min + w / 2.0
        cy = y_min + h / 2.0

        a = np.array([cx, cy])
        b = a + 15.0 * oz
        c = a + 15.0 * oy

        ax.add_line(plt.Line2D([a[0], b[0]], [a[1], b[1]], color="blue"))
        ax.add_line(plt.Line2D([a[0], c[0]], [a[1], c[1]], color="green"))

        ax.text(
            x_min - w / 2,
            y_min,
            f"cls: {int(nms.nmsed_classes[0, i])}\nsco: {nms.nmsed_scores[0, i]:.2f}\nr12:{angle[0]:.2f}\nr23:{angle[1]:.2f}",
            bbox={"facecolor": [0, 1, 0], "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
        # print("x: {:d}, y1: {:d}, w: {:d}, h: {:d}".format(x_min, y_min, w, h))

    plt.savefig(args.o)


args = parser.parse_args()

model = get_efficientdet(args.n, num_classes=args.c, num_anchors=args.a)
model.build(input_shape=(1, IMG_SIZE, IMG_SIZE, 3))
model.load_weights(args.w)

raw_image = tf.io.read_file(args.i)
image = tf.image.decode_image(raw_image, channels=3)

make_prediction(image, score_threshold=0.9)
