# EfficientDet

Repository with my implementation of [EfficientDet](https://arxiv.org/abs/1911.09070).

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"