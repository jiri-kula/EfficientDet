# import datetime
import tensorflow as tf

# import numpy as np
# import keras
# from model.efficientdet import get_efficientdet
# from model.losses import EffDetLoss, AngleLoss
# from model.anchors import SamplesEncoder, Anchors
# from dataset import MyDataset, CSVDataset, image_mosaic, IMG_OUT_SIZE
# from model.utils import to_corners

interpreter = tf.lite.Interpreter("model.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()  # Needed before execution!
input = interpreter.get_input_details()[0]  # Model has single input.
output = interpreter.get_output_details()[0]
input_data = tf.constant(50, shape=[1, 320, 320, 3], dtype="uint8")
interpreter.set_tensor(input["index"], input_data)
interpreter.invoke()
retval = interpreter.get_tensor(output["index"])
retval[0, ...]

print("OK")
