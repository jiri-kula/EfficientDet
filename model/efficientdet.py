"""Implementation of EffDets."""

import tensorflow as tf
from .layers import BiFPN, ClassDetector, BoxRegressor, AngleRegressor
from .backbone import get_backbone
import tensorflow_hub as hub
import re


# https://stackoverflow.com/questions/55428731/how-to-debug-custom-metric-values-in-tf-keras
# @tf.function


class AngleMetric(tf.keras.metrics.Metric):
    def __init__(self, name="angle_alignment", **kwargs):
        super(AngleMetric, self).__init__(name=name, **kwargs)
        self.batch_metric = []
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        class_label = y_true[..., 10]

        angle_labels = y_true[..., 4:10]
        angle_preds = y_pred[..., 4:10]

        loss = angle_labels - angle_preds

        positive_mask = tf.cast(tf.greater(class_label, -1.0), tf.float32)

        r1_pred = angle_preds[..., :3]
        r1_true = angle_labels[..., :3]

        proj1 = tf.reduce_sum(
            tf.multiply(r1_pred, r1_true), -1
        )  # dot prod of true and pred vectors

        r2_pred = angle_preds[..., 3:]
        r2_true = angle_labels[..., 3:]

        proj2 = tf.reduce_sum(
            tf.multiply(r2_pred, r2_true), -1
        )  # dot prod of true and pred vectors

        q = tf.where(
            positive_mask == 1.0, (proj1 + proj2) / 2.0, 0.0
        )  # zero out non-relevat

        # we want all projections to be 1 (fit)

        normalizer = tf.reduce_sum(positive_mask, axis=-1)

        sample_metric = tf.math.divide_no_nan(tf.reduce_sum(q, axis=-1), normalizer)

        batch_metric = tf.reduce_mean(sample_metric, name="angle_sample_metric")
        # self.set_count(self.get_count() + 1)

        # self.batch_metric = (
        #     self.get_metric() * (self.get_count() - 1) + batch_metric
        # ) / self.get_count()

        # self.count += 1
        # a = tf.multiply(self.batch_metric, (self.count - 1))
        # b = tf.add(a, batch_metric)
        # c = self.count
        # self.batch_metric = b / c
        self.batch_metric.append(batch_metric)
        self.count = tf.reduce_mean(self.batch_metric)

    def result(self):
        return self.count

    def get_count(self):
        return self.count

    def set_count(self, val):
        self.count = val

    def get_metric(self):
        return self.count


class EfficientDet(tf.keras.Model):
    """EfficientDet model."""

    def __init__(
        self,
        channels=64,
        num_classes=80,
        num_anchors=9,
        bifpn_depth=3,
        bifpn_kernel_size=3,
        bifpn_depth_multiplier=1,
        bifpn_pooling_strategy="avg",
        heads_depth=3,
        class_kernel_size=3,
        class_depth_multiplier=1,
        box_kernel_size=3,
        box_depth_multiplier=1,
        backbone_name="efficientnet_b0",
        name="efficientdet_d0",
    ):
        """Initialize EffDet. Default args refers to EfficientDet D0.

        Args:
            channels: an integer representing number of units inside each fusing
                node and convolution layer of BiFPN and head models.
            num_classes: an integer representing number of classes to predict.
            num_anchors: an integer representing number of anchor boxes.
            bifpn_depth: an integer representing number of BiFPN layers.
            bifpn_kernel_size: an integer or tuple/list of 2 integers, specifying
                the height and width of the 2D convolution window for BiFPN layers.
            bifpn_depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in BiFPN nodes.
            bifpn_pooling_strategy: a string representing pooling strategy in BiFPN
                layers. 'avg' or 'max'. Otherwise the max pooling will be selected.
            heads_depth: an integer representing number of separable convolutions
                before final convolution in head models.
            class_kernel_size: an integer or tuple/list of 2 integers, specifying
                the height and width of the 2D convolution window for
                classifier model.
            class_depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in classifier model.
            box_kernel_size: an integer or tuple/list of 2 integers, specifying
                the height and width of the 2D convolution window for
                regression model.
            box_depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in regression model.
            name: a string representing model name.
        """
        super().__init__(name=name)

        self.var_freeze_expr = None
        self.loss_tracker = tf.metrics.Mean(name="loss")
        self.angle_metric = AngleMetric()
        self.mean_angle_metric = tf.metrics.Mean(name="mean_angle_metric")

        # self.backbone = get_backbone(backbone_name)
        # self.backbone.trainable = False
        self.backbone = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1",
            trainable=True,
        )

        # self.BiFPN = BiFPN(
        #     channels=channels,
        #     depth=bifpn_depth,
        #     kernel_size=bifpn_kernel_size,
        #     depth_multiplier=bifpn_depth_multiplier,
        #     pooling_strategy=bifpn_pooling_strategy,
        # )

        # self.BiFPN.trainable = True

        self.class_det = ClassDetector(
            channels=channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            depth=heads_depth,
            kernel_size=class_kernel_size,
            depth_multiplier=class_depth_multiplier,
        )
        self.box_reg = BoxRegressor(
            channels=channels,
            num_anchors=num_anchors,
            depth=heads_depth,
            kernel_size=box_kernel_size,
            depth_multiplier=box_depth_multiplier,
        )

        self.angle_reg = AngleRegressor(
            channels=channels,
            num_anchors=num_anchors,
            depth=heads_depth,
            kernel_size=box_kernel_size,
            depth_multiplier=box_depth_multiplier,
        )

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker,
            # self.angle_metric
        ]

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        # features = self.backbone(inputs)
        # features.append(tf.keras.layers.AveragePooling2D()(features[-1]))
        # features.append(tf.keras.layers.AveragePooling2D()(features[-1]))

        # fpn_features = self.BiFPN(features, training=training)

        classes = list()
        boxes = list()
        angles = list()

        c, b = self.backbone(inputs)

        for i in range(0, 5):
            # classes
            tmp1 = self.class_det(c[i], training=training)
            s = tmp1.shape[1]
            tmp = tf.reshape(
                tmp1,
                [batch_size, -1, self.class_det.num_classes],
                # [batch_size, s, s, -1, self.class_det.num_classes],
            )

            # softmaxed_classes = tf.keras.activations.softmax(tmp) # no softmax - let all 0 + background class
            classes.append(tmp)

            # boxes
            tmp1 = self.box_reg(b[i], training=training)
            tmp = tf.reshape(
                tmp1,
                [batch_size, -1, 4],
            )
            boxes.append(tmp)

            # angles
            tmp1 = self.angle_reg(b[i], training=training)
            # tmp = tf.reshape(
            #     tmp1,
            #     [batch_size, -1, 6],  # rotation: r13, r23
            # )
            angles.append(tmp1)

        classes = tf.concat(classes, axis=1)
        classes = tf.keras.activations.sigmoid(
            classes
        )  # uncomment this line before conversin to tflite, but comment out before training

        boxes = tf.concat(boxes, axis=1)
        angles = tf.concat(angles, axis=1)

        retval = tf.concat([boxes, angles, classes], axis=-1)
        return retval

    def _freeze_vars(self):
        if self.var_freeze_expr:
            return [
                v
                for v in self.trainable_variables
                if not re.match(self.var_freeze_expr, v.name)
            ]

    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self._freeze_vars()
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # self.angle_metric.update_state(y_true=y, y_pred=y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        # return {"loss": self.metrics[0], "angle": self.angle_metric.result()}


def get_efficientdet(name="efficientdet_d0", num_classes=80, num_anchors=9):
    models = {
        "efficientdet_d0": (64, 3, 3, "efficientnet_b0"),
        "efficientdet_d1": (88, 4, 3, "efficientnet_b1"),
        "efficientdet_d2": (112, 5, 3, "efficientnet_b2"),
        "efficientdet_d3": (160, 6, 4, "efficientnet_b3"),
        "efficientdet_d4": (224, 7, 4, "efficientnet_b4"),
        "efficientdet_d5": (288, 7, 4, "efficientnet_b5"),
        "efficientdet_d6": (384, 8, 5, "efficientnet_b6"),
        "efficientdet_d7": (384, 8, 5, "efficientnet_b7"),
    }
    return EfficientDet(
        channels=models[name][0],
        num_classes=num_classes,
        num_anchors=num_anchors,
        bifpn_depth=models[name][1],
        heads_depth=models[name][2],
        name=name,
    )
