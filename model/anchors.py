"""Implementation of anchor boxes generator and encoder of training data."""

import tensorflow as tf
import sys
from .utils import compute_iou

INPUT_SIZE = 320


class Anchors:
    """Anchor boxes generator."""

    def __init__(
        self,
        aspect_ratios=[0.5, 1.0, 2.0],  # wide rectangle  # square  # tall rectangle
        scales=[
            2.0 / 3.0,
            1.0,
            4.0 / 3.0,
        ],  # WARNING: change of scales or aspect_ratios -> modify train.py num_anchors=3 parameter of the model
    ):
        """Initialize anchors generator.

        Args:
            aspect_ratios: a list of floats representing aspect
                ratios of anchor boxes on each feature level.
            scales: a list of floats representing different scales
                of anchor boxes on each feature level.
        """
        self._aspect_ratios = aspect_ratios
        self._scales = [1.0 * scale for scale in scales]
        self._num_anchors = len(aspect_ratios) * len(scales)

        self._strides = [2**i for i in range(3, 8)]
        # self._areas = [i**2 for i in [24.0, 48.0, 96.0, 192.0, 384.0]]
        start_value = 16.0
        length = 5
        v = [start_value * (2**i) for i in range(length)]
        self._areas = [i**2 for i in v]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Compute height and width for each anchor box on each level.

        Returns:
            A float tensor with shape (5, num_anchors, 2) where each
                pair representing height and width of anchor box.
        """
        all_dims = list()
        for area in self._areas:
            level_dims = list()
            for aspect_ratio in self._aspect_ratios:
                height = tf.math.sqrt(area) * aspect_ratio
                width = area / height
                dims = tf.cast([width, height], tf.float32)
                for scale in self._scales:
                    level_dims.append(dims * scale)
            all_dims.append(tf.stack(level_dims, axis=0))
        return tf.stack(all_dims, axis=0)

    @tf.function
    def _get_anchors(self, feature_height, feature_width, level):
        """Get anchors for with given height and width on given level.

        Args:
            feature_height: an integer representing height of feature map.
                Should be divisible by 2**level.
            feature_width: an integer representing width of feature map.
                Should be divisible by 2**level.
            level: an integer from range [3, 7] representing level
                of feature map.
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        xs = tf.tile(
            tf.reshape(rx, [1, -1]), [tf.shape(ry)[0], 1]
        )  # this is like repmat, or meshgrid with the next line
        ys = tf.tile(
            tf.reshape(ry, [-1, 1]), [1, tf.shape(rx)[0]]
        )  # at feature dims (40, 20, 10, 5, 3, ...)

        centers = (
            tf.stack([xs, ys], axis=-1) * self._strides[level - 3]
        )  # transfers to image dims (320)
        centers = tf.reshape(centers, [-1, 1, 2])
        centers = tf.tile(centers, [1, self._num_anchors, 1])
        centers = tf.reshape(centers, [-1, 2])

        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height * feature_width, 1]
        )
        return tf.concat([centers, dims], axis=-1)

    @tf.function
    def get_anchors(self, image_height, image_width):
        """Get anchors for given height and width on all levels.

        Args:
            image_height: an integer representing height of image.
            image_width: an integer representing width of image.
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2**i),
                tf.math.ceil(image_width / 2**i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


class SamplesEncoder:
    """Enchoder of training batches."""

    def __init__(
        self,
    ):
        self._anchors = Anchors()
        self._box_variance = tf.cast([0.1, 0.1, 0.2, 0.2], tf.float32)
        self.anchor_boxes = None

    @tf.function
    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Assign ground truth boxes to all anchor boxes."""

        iou = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou, axis=1)
        matched_gt_idx = tf.argmax(iou, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)

        # assert len(tf.where(positive_mask)) > 0

        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    # @tf.autograph.experimental.do_not_convert
    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    # @tf.autograph.experimental.do_not_convert
    @tf.function
    def _encode_sample(self, image_shape, gt_boxes, classes, angles):
        if self.anchor_boxes is None:
            self.anchor_boxes = self._anchors.get_anchors(
                image_shape[0], image_shape[1]
            )

        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            self.anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(
            gt_boxes, matched_gt_idx
        )  # select one box from gt_boxes for each anchor
        box_target = self._compute_box_target(
            self.anchor_boxes, matched_gt_boxes
        )  # compute shift + scale of anchor to match 'gt box'
        tf.debugging.check_numerics(box_target, "box_target contains NaN or Inf")

        # tf.print("classes:", classes, output_stream=sys.stderr)
        # tf.print("angles:", angles, output_stream=sys.stderr)
        matched_gt_classes = tf.gather(classes, matched_gt_idx)
        class_target = tf.where(
            tf.equal(positive_mask, 1.0), (matched_gt_classes), -1.0
        )
        class_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, class_target)
        class_target = tf.expand_dims(class_target, axis=-1)

        matched_gt_angles = tf.gather(angles, matched_gt_idx)
        # tf.print("matched_gt_angles:", matched_gt_angles, output_stream=sys.stderr)

        # angle_target = tf.where(tf.equal(positive_mask, 1.0), matched_gt_angles, -1.0)
        # angle_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, class_target)
        # angle_target = tf.expand_dims(class_target, axis=-1)

        label = tf.concat([box_target, matched_gt_angles, class_target], axis=-1)

        # if len(classes) > 0:
        #     assert (
        #         len(tf.where(class_target[:, 0] > -1.0)) > 0
        #     ), "No anchor matched to gt_boxes"

        return label

    # @tf.autograph.experimental.do_not_convert
    @tf.function
    def encode_batch(self, images, gt_boxes, classes, angles):
        """Encode batch for training."""

        images_shape = tf.shape(images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size)
        for i in range(batch_size):
            label = self._encode_sample(
                images_shape, gt_boxes[i], classes[i], angles[i]
            )
            labels = labels.write(i, label)
        images = tf.keras.applications.efficientnet.preprocess_input(images)
        return images, labels.stack()
