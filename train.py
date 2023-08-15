# %%
"""Script for creating and training a new model."""

import tensorflow as tf
import numpy as np
import keras
from model.efficientdet import get_efficientdet
from model.losses import EffDetLoss
from model.anchors import SamplesEncoder, Anchors
from dataset import MyDataset, CSVDataset, image_mosaic, IMG_OUT_SIZE
from model.utils import to_corners


MODEL_NAME = "efficientdet_d0"

NUM_CLASSES = 6

EPOCHS = 1000
EAGERLY = False
BATCH_SIZE = 16 if EAGERLY else 64

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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    run_eagerly=EAGERLY,
)

# %%

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
meta_train = "/mnt/c/Edwards/Output/RV12-lic/meta.csv"
meta_test = "/mnt/c/Edwards/Output/RV12-lic/meta_test.csv"
train_data = CSVDataset(meta_train, None, BATCH_SIZE)
test_data = CSVDataset(meta_test, None, BATCH_SIZE)


# model.build(input_shape=(BATCH_SIZE, 320, 320, 3))
model.summary(show_trainable=True)

current_epoch = 1411

model.load_weights('tmp/fit/epoch_{:d}'.format(current_epoch))

class CustomCallback(keras.callbacks.Callback):
    # def on_train_batch_end(self, batch, logs=None):
    #     images, lbl = train_data.__getitem__(batch)
    #     # print(images.shape)
    #     image_mosaic(images)

    def on_epoch_end(self, epoch, logs=None):
        global current_epoch
        current_epoch += 1
        model.save_weights("tmp/fit/epoch_{:d}".format(current_epoch))


def shared_mem_multiprocessing(sequence, workers=2, queue_max_size=16):
    from multiprocessing import Process, Queue, shared_memory, managers

    queue = Queue(maxsize=queue_max_size)
    manager = managers.SharedMemoryManager()
    manager.start()

    def worker(sequence, idxs):
        for i in idxs:
            x = sequence[i]

            shm = manager.SharedMemory(size=x.nbytes)
            a = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf, offset=0)

            a[:] = x[:]
            queue.put((a.shape, a.dtype, shm.name))

    idxs = np.array_split(np.arange(len(sequence)), workers)
    args = zip([sequence] * workers, idxs)
    processes = [Process(target=worker, args=(s, i)) for s, i in args]
    _ = [p.start() for p in processes]

    for i in range(len(sequence)):
        x_shape, x_dtype, shm_name = queue.get(block=True)
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        x = np.ndarray(x_shape, dtype=x_dtype, buffer=existing_shm.buf, offset=0)
        yield x


# %%
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# gen = shared_mem_multiprocessing(train_data, workers=4)

history = model.fit(
    train_data,
    epochs=EPOCHS,
    workers=1,
    use_multiprocessing=False,
    validation_data=test_data,
    callbacks=[CustomCallback()],
)

# %%
class PostProcessedEfficientDet(tf.keras.Model):
  def __init__(self, efficient_det):
    super().__init__(name="PostProcessedEfficientDet")

    # self.box_variance = tf.cast([0.1, 0.1, 0.2, 0.2], tf.float32)
    # an = Anchors()
    # self.anchor_boxes = an.get_anchors(IMG_OUT_SIZE, IMG_OUT_SIZE)
    self.efficient_det = efficient_det

  def call(self, inputs):
    preds = self.efficient_det(inputs)
        
    boxes = preds[..., :4]
    # boxes = preds[..., :4] * self.box_variance

    # boxes = tf.concat(
    #     [
    #         boxes[..., :2] * self.anchor_boxes[..., 2:] + self.anchor_boxes[..., :2],
    #         tf.exp(boxes[..., 2:]) * self.anchor_boxes[..., 2:],
    #     ],
    #     axis=-1,
    # )
    
    # boxes = to_corners(boxes)
    angles = preds[..., 4:6]
    classes = tf.nn.sigmoid(preds[..., 6:])

    return boxes, classes, angles

    # nms = tf.image.combined_non_max_suppression(
    #     tf.expand_dims(boxes, axis=2),
    #     classes,
    #     max_output_size_per_class=4,
    #     max_total_size=8,
    #     iou_threshold=0.5,
    #     score_threshold=float('-inf'),
    #     clip_boxes=False,
    # )

    # return nms.valid_detections, nms.nmsed_boxes, nms.nmsed_scores, nms.nmsed_classes

post_model = PostProcessedEfficientDet(model)

post_model.compute_output_shape((1, 320, 320, 3))

#%%
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 320, 320, 3)
        yield [data.astype(np.float32)]


# Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model') # path to the SavedModel directory
converter = tf.lite.TFLiteConverter.from_keras_model(post_model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_dataset
# This ensures that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
# converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.experimental_new_converter = False
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
# converter.experimental_new_converter = False
tflite_model = converter.convert()

# Save the model.
with open("post_model.tflite", "wb") as f:
    f.write(tflite_model)
# %%

interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()  # Needed before execution!
input = interpreter.get_input_details()[0]  # Model has single input.
output = interpreter.get_output_details()[0]
input_data = tf.constant(50, shape=[1, 320, 320, 3], dtype="uint8")
interpreter.set_tensor(input["index"], input_data)
interpreter.invoke()
retval = interpreter.get_tensor(output["index"])
retval[0, ...]


# %%
model.predict(images[0])
# %%
