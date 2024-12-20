"""Inference script."""

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model.anchors import Anchors
from model.utils import to_corners, resize_and_pad
from model.efficientdet import EfficientDet
from tfrecord_decode import decode_fn, decode_raw, raw2label

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

    boxes, angles, classes = model.predict(tf.expand_dims(padded_image, axis=0))

    boxes = boxes * box_variance
    boxes = tf.concat(
        [
            boxes[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
            tf.exp(boxes[..., 2:]) * anchor_boxes[..., 2:],
        ],
        axis=-1,
    )
    boxes = to_corners(boxes)
    classes = tf.nn.sigmoid(classes)

    valid_dets = 0
    while valid_dets < 2 and score_threshold > 0:
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

        score_threshold -= 0.01

    max_anchor_scores = tf.reduce_max(classes, axis=-1)

    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()

    for i in range(0, min(10, valid_dets)):
        x_min, y_min, x_max, y_max = nms.nmsed_boxes[0, i] / scale
        w, h = x_max - x_min, y_max - y_min
        # x_min, y_min, w, h = 75, 40, 35, 20
        patch = plt.Rectangle(
            [x_min, y_min], w, h, fill=False, edgecolor=[0, 1, 0], linewidth=1
        )
        ax.add_patch(patch)

        angle_idxes = tf.where(max_anchor_scores[0] == nms.nmsed_scores[0, i])
        # if len(angle_idx) == 0:
        #     continue
        # angle_idx = angle_idx[0]
        for angle_idx in angle_idxes:
            angle = angles[0, int(angle_idx)]

            r1 = angle[:3]
            r2 = angle[3:]
            r3 = np.cross(r1, r2)

            cRo = np.stack([r1, r2, r3], axis=1)

            s = 15.0

            xo = np.array([1, 0, 0])
            yo = np.array([0, 1, 0])
            zo = np.array([0, 0, 1])

            xc = np.dot(cRo, xo)
            yc = np.dot(cRo, yo)
            zc = np.dot(cRo, zo)

            cx = x_min + w / 2.0
            cy = y_min + h / 2.0

            a = np.array([cx, cy, 0])
            b = a + s * xc
            c = a + s * yc
            d = a + s * zc

            ax.add_line(plt.Line2D([a[0], c[0]], [a[1], c[1]], color="green"))
            ax.add_line(plt.Line2D([a[0], d[0]], [a[1], d[1]], color="blue"))
            ax.add_line(
                plt.Line2D([a[0], b[0]], [a[1], b[1]], color="red")
            )  # last to become visible

            ax.text(
                x_min - w / 2,
                y_min - h / 2,
                f"cls: {int(nms.nmsed_classes[0, i])}\nsco: {nms.nmsed_scores[0, i]:.2f}\nr31:{r1[2]:.2f}",
                bbox={"facecolor": [0, 1, 0], "alpha": 0.2},
                clip_box=ax.clipbox,
                clip_on=True,
            )
            # print("x: {:d}, y1: {:d}, w: {:d}, h: {:d}".format(x_min, y_min, w, h))

    plt.savefig(args.o)


args = parser.parse_args()

model = EfficientDet(
    channels=64,
    num_classes=3,
    num_anchors=9,
    bifpn_depth=3,
    heads_depth=3,
    name="efficientdet_d0",
    export_tflite=False,
)

model.var_freeze_expr = (
    "efficientnet-lite0|resample_p6"  # "efficientnet-lite0|resample_p6|fpn_cells"
)
model.build(input_shape=(1, IMG_SIZE, IMG_SIZE, 3))
model.load_weights(args.w)

# raw_image = tf.io.read_file(args.i)
# image = tf.image.decode_image(raw_image, channels=3)

example_path = "/home/jiri/winpart/Edwards/tfrecords/_home_jiri_remote_sd_DetectionData_Dataset_zaznamy_z_vyroby_2023_03_03_rv12_09_18_26.tfrecord"
example_path = "/home/jiri/winpart/Edwards/tfrecords/synth_6.tfrecord"
example_path = "/home/jiri/winpart/Edwards/tfrecords_allrot/_home_jiri_remote_sd_DetectionData_Dataset_zaznamy_z_vyroby_2023_03_08_rv12_09_47_27.tfrecord"

ds = tf.data.TFRecordDataset([example_path]).skip(1).take(1)
sample = decode_fn(next(iter(ds)))
image = tf.cast(sample[0], tf.uint8)

make_prediction(image, score_threshold=1.0)
