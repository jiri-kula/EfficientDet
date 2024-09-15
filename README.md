# Python
wget https://www.python.org/ftp/python/3.9.11/Python-3.9.11.tar.xz
xz -d Python-3.9.11.tar.xz
tar -xvf Python-3.9.11.tar
cd Python-3.9.11
sudo apt install        \
      libffi-dev        \
      libgl1            \
      libsqlite3-dev    \
      libssl-dev        \
      -y
./configure --enable-optimizations
make -j12
sudo make altinstall

# EfficientDet

Repository with my implementation of [EfficientDet](https://arxiv.org/abs/1911.09070).

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
CUPTI_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib/
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUPTI_PATH:$LD_LIBRARY_PATH
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Ubuntu
sudo sh cuda_12.2.2_535.104.05_linux.run

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-12.2/

Please make sure that
 -   PATH includes /usr/local/cuda-12.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.2/lib64, or, add /usr/local/cuda-12.2/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.2/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 535.00 is required for CUDA 12.2 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver
# Training checklist
1. Check your data annotation `datacheck.py`
2. Measure how to set anchor boxes `anchor_histogram.py`, modify `model/anchors.py` at *aspects* and *areas*.

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

2) Do post processing in custom c++ code as we did that for palm and landmark detection. Anchors need to be generated in c++ and kept in sync with python training code manually; or save them to anchors.txt

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

# Mount windows partition into Ubuntu
```
mkdir -p ~/winpart
```

```
jiri@jiri-81Y6:~/EfficientDet$ lsblk
NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0         7:0    0     4K  1 loop /snap/bare/5
loop1         7:1    0  55.7M  1 loop /snap/core18/2785
loop2         7:2    0  55.7M  1 loop /snap/core18/2790
loop3         7:3    0  63.4M  1 loop /snap/core20/1974
loop4         7:4    0  63.5M  1 loop /snap/core20/2015
loop5         7:5    0  73.9M  1 loop /snap/core22/858
loop6         7:6    0  73.9M  1 loop /snap/core22/864
loop7         7:7    0 240.5M  1 loop /snap/firefox/3206
loop8         7:8    0 238.8M  1 loop /snap/firefox/3252
loop9         7:9    0 218.4M  1 loop /snap/gnome-3-34-1804/93
loop10        7:10   0 485.5M  1 loop /snap/gnome-42-2204/126
loop11        7:11   0   497M  1 loop /snap/gnome-42-2204/141
loop12        7:12   0  91.7M  1 loop /snap/gtk-common-themes/1535
loop13        7:13   0 115.7M  1 loop /snap/slack/105
loop14        7:14   0 113.3M  1 loop /snap/slack/89
loop15        7:15   0  12.3M  1 loop /snap/snap-store/959
loop16        7:16   0  40.8M  1 loop /snap/snapd/20092
loop17        7:17   0  40.9M  1 loop /snap/snapd/20290
loop18        7:18   0   452K  1 loop /snap/snapd-desktop-integration/83
loop19        7:19   0 320.4M  1 loop /snap/vlc/3078
nvme1n1     259:0    0   1.8T  0 disk 
├─nvme1n1p1 259:1    0   100M  0 part 
├─nvme1n1p2 259:2    0    16M  0 part 
├─nvme1n1p3 259:3    0   1.8T  0 part <<=== for example this big drive on Legion5 of (jiri)
└─nvme1n1p4 259:4    0   530M  0 part 
nvme0n1     259:5    0 465.8G  0 disk 
├─nvme0n1p1 259:6    0 209.8G  0 part 
├─nvme0n1p2 259:7    0     1G  0 part /boot/efi
└─nvme0n1p3 259:8    0   255G  0 part /var/snap/firefox/common/host-hunspell
```

1) read-only access
```
sudo mount -t ntfs -o ro /dev/nvme1n1p3 ~/winpart
```

2) read-write access
```
sudo mount -t ntfs /dev/nvme1n1p3 ~/winpart
```

## EfficientDet properties
EfficientDet is a family of object detection models that are designed to be both scalable and efficient. 

Here's a breakdown of the key components you mentioned and how they work together:

### Aspect Ratios
Aspect ratios define the shape of the anchors (bounding boxes) used for detecting objects. EfficientDet typically uses aspect ratios like 1:1, 1:2, and 2:1. These ratios help the model detect objects of different shapes and sizes.

### Scales
Scales determine the size of the anchors. EfficientDet uses scales like (2^0), (2^{1/3}), and (2^{2/3}). These scales allow the model to detect objects at various sizes within the same aspect ratio.

### Levels
Levels refer to the different layers in the feature pyramid network (FPN) used in EfficientDet. Each level corresponds to a different resolution of the input image, allowing the model to detect objects at multiple scales. For example, lower levels might detect larger objects, while higher levels detect smaller objects.

### Strides 
Strides are the steps taken in the feature map to generate anchors. For instance, a stride of 8 means that the anchors are generated every 8 pixels in the feature map. EfficientDet uses strides like 8, 16, 32, 64, and 128, corresponding to different levels in the FPN.

### Number of Anchors
The number of anchors is determined by the combination of aspect ratios and scales. For example, if you have 3 aspect ratios and 3 scales, you will have (3 \times 3 = 9) anchors per location in the feature map.

### How They Work Together - Anchor Generation: 
Anchors are generated at each level of the FPN using the specified aspect ratios and scales. This results in a set of anchors that can detect objects of various shapes and sizes.
- Feature Extraction: The FPN extracts features from the input image at different resolutions, corresponding to different levels.
- Anchor Matching: During training, each anchor is matched with the ground truth bounding boxes. The model learns to adjust the anchors to better fit the objects.
- Prediction: During inference, the model predicts the class and bounding box adjustments for each anchor. The anchors are then adjusted to form the final bounding boxes.
EfficientDet's design, including the use of aspect ratios, scales, levels, strides, and the number of anchors, allows it to efficiently and accurately detect objects at multiple scales and resolutions.
If you have any more questions or need further clarification, feel free to ask!
: EfficientDet: Scalable and Efficient Object Detection
: EfficientDet/utils/anchors.py at master - GitHub

## Known issues
- Access is denied because the NTFS volume is already exclusively opened
use `lsblk` to check if the partition is already mounted, if so use

```
sudo umount /media/jiri/D6667DDE667DBFB3 (or whatever after /jiri/...)
```