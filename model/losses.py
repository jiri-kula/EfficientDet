"""EfficientDet losses

This script contains implementations of Focal loss for classification task and
Huber loss for regression task. Also script includes composition of these losses
for quick setup of training pipeline.
"""

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss implementations."""

    def __init__(self, alpha=0.25, gamma=1.5, label_smoothing=0.1, name="focal_loss"):
        """Initialize parameters for Focal loss.

        FL = - alpha_t * (1 - p_t) ** gamma * log(p_t)
        This implementation also includes label smoothing for preventing overconfidence.
        """
        super().__init__(name=name, reduction="none")
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """Calculate Focal loss.

        Args:
            y_true: a tensor of ground truth values with
                shape (batch_size, num_anchor_boxes, num_classes).
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).

        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        prob = tf.sigmoid(y_pred)
        # prob = y_pred
        pt = y_true * prob + (1 - y_true) * (1 - prob)
        at = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        loss = at * (1.0 - pt) ** self.gamma * ce
        return tf.reduce_sum(loss, axis=-1)


class BoxLoss(tf.keras.losses.Loss):
    """Huber loss implementation."""

    def __init__(self, delta=1.0, name="box_loss"):
        super().__init__(name=name, reduction="none")
        self.delta = delta

    def call(self, y_true, y_pred):
        """Calculate Huber loss.

        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 4).
            y_pred: a tensor of predicted values with shape (batch_size, num_anchor_boxes, 4).

        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        loss = tf.abs(y_true - y_pred)
        l1 = self.delta * (loss - 0.5 * self.delta)
        l2 = 0.5 * loss**2
        box_loss = tf.where(tf.less(loss, self.delta), l2, l1)
        loss = tf.reduce_sum(box_loss, axis=-1)
        return loss


class AngleLoss(tf.keras.losses.Loss):
    """SAE implementation."""

    def __init__(self, delta=1.0, name="angle_loss"):
        super().__init__(name=name, reduction="none")

    def call(self, angle_labels, angle_preds):
        num_anchors = angle_labels.shape[1]

        # construct Predicted rotation matrix
        r1_pred = angle_preds[..., :3]
        r2_pred = angle_preds[..., 3:]
        r3_pred = tf.linalg.cross(r1_pred, r2_pred)

        P = tf.concat([r1_pred, r2_pred, r3_pred], axis=-1)
        P = tf.reshape(P, (-1, num_anchors, 3, 3))
        P = tf.transpose(P, perm=[0, 1, 3, 2])  # so that r1, r2, r3 form columns

        # construct Labeled rotation matrix
        r1_true = angle_labels[..., :3]
        r2_true = angle_labels[..., 3:]
        r3_true = tf.linalg.cross(r1_true, r2_true)
        L = tf.concat([r1_true, r2_true, r3_true], axis=-1)
        L = tf.reshape(L, (-1, num_anchors, 3, 3))
        L = tf.transpose(L, perm=[0, 1, 3, 2])  # so that r1, r2, r3 form columns

        # difference matrix between Predicted and Labeled rotation
        LT = tf.transpose(L, perm=[0, 1, 3, 2])  # TODO: remove double transpose
        Q = P @ LT  # checked ok as np.dot(P, LT)

        # we want the angle between rotations to be zero
        TrQ = (tf.linalg.trace(Q) - 1.0) / 2.0

        # numerator = (TrQ - 1) / 2.0
        # numerator = tf.where(numerator > 1.0, 1.0, numerator)
        # numerator = tf.where(numerator < -1.0, -1.0, numerator)
        TrQ = tf.clip_by_value(TrQ, clip_value_min=-0.99, clip_value_max=0.99)

        theta = tf.acos(TrQ)  # [0, pi]

        # axis of difference rotation not computed here

        # Reference:
        # [1] https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
        # [2] https://math.stackexchange.com/questions/744736/rotation-matrix-to-axis-angle

        return theta


class EffDetLoss(tf.keras.losses.Loss):
    """Composition of Focal and Huber losses."""

    def __init__(
        self,
        num_classes=80,
        alpha=0.25,
        gamma=1.5,
        label_smoothing=0.1,
        delta=1.0,
        name="effdet_loss",
    ):
        """Initialize Focal and Huber loss.

        Args:
            num_classes: an integer number representing number of
                all possible classes in training dataset.
            alpha: a float number for Focal loss formula.
            gamma: a float number for Focal loss formula.
            label_smoothing: a float number of label smoothing intensity.
            delta: a float number representing a threshold in Huber loss
                for choosing between linear and cubic loss.
        """
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.class_loss = FocalLoss(
            alpha=alpha, gamma=gamma, label_smoothing=label_smoothing
        )
        self.box_loss = BoxLoss(delta=delta)
        self.angle_loss = tf.keras.losses.MeanSquaredError()
        self.num_classes = num_classes

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        """Calculate Focal and Huber losses for every anchor box.

        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 5)
                representing anchor box correction and class label.
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).

        Returns:
            loss: a float loss value.
        """
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        box_labels = y_true[..., :4]
        box_preds = y_pred[..., :4]

        # angle_labels = tf.expand_dims(y_true[..., 4:6], -1)
        # angle_preds =  tf.expand_dims(y_pred[..., 4:6], -1)

        angle_labels = y_true[..., 4:10]
        angle_preds = y_pred[..., 4:10]

        cls_labels = tf.one_hot(
            tf.cast(y_true[..., 10], dtype=tf.int32),
            depth=self.num_classes,
            dtype=tf.float32,
        )
        cls_preds = y_pred[..., 10:]

        positive_mask = tf.cast(tf.greater(y_true[..., 10], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[..., 10], -2.0), dtype=tf.float32)

        clf_loss = self.class_loss(cls_labels, cls_preds)
        box_loss = self.box_loss(box_labels, box_preds)
        ang_loss = self.angle_loss(angle_labels, angle_preds)

        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        ang_loss = tf.where(
            tf.equal(positive_mask, 1.0), ang_loss, 0.0
        )  # TODO: decide which mask to use

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        ang_loss = tf.math.divide_no_nan(tf.reduce_sum(ang_loss, axis=-1), normalizer)

        # loss = clf_loss + box_loss + ang_loss

        retval = tf.reduce_mean(tf.stack([box_loss, ang_loss, clf_loss]), axis=-1)

        return retval
