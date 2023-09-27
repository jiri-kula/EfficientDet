# EfficientDet

Repository with my implementation of [EfficientDet](https://arxiv.org/abs/1911.09070).

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
CUPTI_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib/
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUPTI_PATH:$LD_LIBRARY_PATH
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# CustomOp
Object detection z tflite model maker: síť má na konci vrstvu `TFLite_Detection_PostProcess`, která je build-in v tflite. Podívej se na `tensorflow/tensorflow/lite/kernels/register.cc`.

`TFLite_Detection_PostProcess` je zřejmně kompatibilní s EdgeTPU compiler, ale neobsahuje extrakci sin/cosine, kterou potřebuji pro rotaci blade v Edwards.

Sít je typu SSD. Na výstupu má surové
[x1, x2, y1, y2, sin, cos, class logits], kde x, y, jsou offset a scale Anchorů a class logits je také třeba dále zpravovat na třídu a score.

Existují dvě strategie:
1) Add post processing layer like `TFLite_Detection_PostProcess`, but it lacks angle. Adding custom-op is not supported in edgetpu_compiler resulting to error

```
Edge TPU Compiler version 16.0.384591198
Started a compilation timeout timer of 180 seconds.
ERROR: Encountered unresolved custom op: Atan.
ERROR: Node number 1 (Atan) failed to prepare.

Compilation failed: Model failed in Tflite interpreter. Please ensure model can be loaded/run in Tflite interpreter.
Compilation child process completed within timeout period.
Compilation failed! 
```

2) Do post processing in custom c++ code as we did that for palm and landmark detection. Anchors need to be generated in c++ and kept in sync with python training code manually.

# Reference
[Better rotation representations for accurate pose estimation](https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f)

# Inspiration
By tflite model maker
`/home/jiri/custom_model_maker/examples/tensorflow_examples/lite/model_maker/pip_package/src/tensorflow_examples/lite/model_maker/third_party/efficientdet/keras/postprocess.py`

See comments in `def postprocess_tflite(params, cls_outputs, box_outputs):`

https://keras.io/examples/vision/retinanet/

# Rotation
/home/jiri/EfficientDet/model/rotation.py in line 7
      40 # %%
      41 # see
      42 # [1] https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
      43 # [2] https://math.stackexchange.com/questions/744736/rotation-matrix-to-axis-angle
      45 a = 10.0 / 180.0 * 3.14
----> 46 R = Rx(0.0) @ Ry(0.0) @ Rz(0.0)  # if label rotation
      47 Q = Rx(0.0) @ Ry(0.0) @ Rz(a)  # and predicted rotation differ by angle a
     49 QT = tf.transpose(Q)