"""Implementation of EffDets."""

import tensorflow as tf
from .layers import BiFPN, ClassDetector, BoxRegressor, AngleRegressor
from .backbone import get_backbone
import tensorflow_hub as hub
import re
from .losses import BoxLoss, AngleLoss, FocalLoss


# https://stackoverflow.com/questions/55428731/how-to-debug-custom-metric-values-in-tf-keras
# @tf.function


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
        export_tflite=False,
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
        self.box_tracker = tf.metrics.Mean(name="box")
        self.angle_tracker = tf.metrics.Mean(name="angle")
        self.class_tracker = tf.metrics.Mean(name="class")

        # self.loss_comp = EffDetLoss(num_classes=3)

        delta = 1.0
        self.box_loss = BoxLoss(delta=delta)
        # self.angle_loss = tf.losses.
        # AngleLoss(delta=delta)
        self.class_loss = FocalLoss(alpha=0.25, gamma=1.5, label_smoothing=0.1)

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
            channels=2 * 64,
            num_anchors=num_anchors,
            depth=heads_depth,
            kernel_size=box_kernel_size,
            depth_multiplier=box_depth_multiplier,
        )

        self.export_tflite = export_tflite

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker,
            self.box_tracker,
            self.angle_tracker,
            self.class_tracker,
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
            tmp1 = self.angle_reg(tf.concat([b[i], c[i]], axis=-1), training=training)
            # tmp = tf.reshape(
            #     tmp1,
            #     [batch_size, -1, 6],  # rotation: r13, r23
            # )
            angles.append(tmp1)

        classes = tf.concat(classes, axis=1)

        boxes = tf.concat(boxes, axis=1)
        angles = tf.concat(angles, axis=1)

        if self.export_tflite:
            # apply sigmoid transform on class predicitons
            classes = tf.keras.activations.sigmoid(
                classes
            )  # uncomment this line before conversin to tflite, but comment out before training

            # zero out boxes to check quantization of our model without boxes
            # boxes = tf.where(tf.equal(boxes, 1.0), 0.0, 0.0)
        # retval = tf.concat([boxes, angles, classes], axis=-1)
        # return retval
        return boxes, angles, classes

    def _freeze_vars(self, var_freeze_expr):
        return [
            v for v in self.trainable_variables if not re.match(var_freeze_expr, v.name)
        ]

    # @tf.autograph.experimental.do_not_convert
    # @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y_true = data

        with tf.GradientTape() as tape:
            box_preds, angle_preds, cls_preds = self(x, training=True)  # Forward pass

            # extract the three labels
            box_labels = y_true[..., :4]
            angle_labels = y_true[..., 4:10]
            cls_labels = tf.one_hot(
                tf.cast(y_true[..., 10], dtype=tf.int32),
                depth=self.class_det.num_classes,
                dtype=tf.float32,
            )

            # filter anchor boxes
            positive_mask = tf.greater(y_true[..., 10], -1.0)
            ignore_mask = tf.cast(tf.equal(y_true[..., 10], -2.0), dtype=tf.float32)

            angle_positive_mask = tf.greater(tf.reduce_sum(tf.abs(angle_labels)), 0.0)
            angle_positive_mask = tf.math.logical_and(
                positive_mask, angle_positive_mask
            )

            positive_mask = tf.cast(positive_mask, dtype=tf.float32)
            angle_positive_mask = tf.cast(angle_positive_mask, dtype=tf.float32)

            # loss for each anchor
            clf_loss = self.class_loss(cls_labels, cls_preds)
            box_loss = self.box_loss(box_labels, box_preds)
            # ang_loss = self.angle_loss(angle_labels, angle_preds)
            ang_loss = tf.losses.MAE(angle_preds, angle_preds)

            # zero out irrelevant anchors
            clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
            box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

            ang_loss = tf.where(tf.equal(angle_positive_mask, 1.0), ang_loss, 0.0)
            # ang_loss = tf.where(# zero out angle loss where label does not carry it (all zeros)
            #     tf.equal(angle_ignore_mask, 1.0), 0.0, ang_loss
            # )

            # average loss across samples so that there remains a scalar loss for each batch
            normalizer = tf.reduce_sum(positive_mask, axis=-1)
            clf_loss = tf.math.divide_no_nan(
                tf.reduce_sum(clf_loss, axis=-1), normalizer
            )
            box_loss = tf.math.divide_no_nan(
                tf.reduce_sum(box_loss, axis=-1), normalizer
            )

            normalizer = tf.reduce_sum(angle_positive_mask, axis=-1)
            ang_loss = tf.math.divide_no_nan(
                tf.reduce_sum(ang_loss, axis=-1), normalizer
            )

            # average loss across batches so that remains a scalar loss for each (box, angle, class)
            losses = tf.reduce_mean(
                tf.stack(
                    [
                        box_loss,
                        ang_loss,
                        clf_loss,
                    ]
                ),
                axis=-1,
            )

            # let total loss be a sum of particular losses = box + angle + class
            loss = tf.reduce_sum(losses)

            # have_angles = tf.reduce_all(tf.equal(positive_mask, angle_positive_mask))
        # Compute gradients
        if self.var_freeze_expr is not None:
            trainable_vars = self._freeze_vars(self.var_freeze_expr)
            # if have_angles:
            #     trainable_vars = self._freeze_vars(tf.strings.join([self.var_freeze_expr, '|angle_regressor']))
            # else:
            #     trainable_vars = self._freeze_vars(self.var_freeze_expr)
        else:
            trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                if metric.name == "box":
                    metric.update_state(losses[0])
                elif metric.name == "angle":
                    metric.update_state(losses[1])
                elif metric.name == "class":
                    metric.update_state(losses[2])
                # metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        # return {"loss": self.metrics[0], "angle": self.angle_metric.result()}
