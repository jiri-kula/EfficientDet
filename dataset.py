import keras
import numpy as np
import tensorflow as tf
from model.anchors import SamplesEncoder

IMG_SIZE = 320


class MyDataset(keras.utils.Sequence):
    def __init__(self, data_dir, aug, batch_size, train=True):
        self.data_dir = data_dir
        self.aug = aug
        self.batch_size = batch_size
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        # return len(self.data_dir) // self.batch_size
        return 1000

    def __getitem__(self, index):
        # indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # image_keys_temp = [self.data_dir[k] for k in indexes]
        # (images, keypoints) = self.__data_generation(image_keys_temp)

        train_images = []
        lbl_boxes = []
        lbl_classes = []

        Image = tf.keras.utils.load_img(self.data_dir + "/images/file_2586.png")
        Image = Image.resize((256, 256))

        Image = tf.keras.utils.img_to_array(Image)
        # each box is of the format [x, y, width, height]
        BoundingBoxes = np.array(
            [[4.0, 4.0, 22.627417, 45.254833], [30.0, 30.0, 22.627417, 45.254833]]
        ).astype(
            np.single
        )  # center, dimension
        # ClassLabels = np.zeros((1, 80)).astype(np.single)
        # ClassLabels[0][0] = 1

        # y = np.zeros((self.batch_size, 12276, 84))

        # Image = np.zeros((self.batch_size, *Image.shape))
        # BoundingBoxes = np.zeros((self.batch_size, 4), dtype=int)
        # ClassLabels = np.zeros((self.batch_size, 1), dtype=int)

        for b in range(0, self.batch_size):
            train_images.append(Image)
            lbl_boxes.append(BoundingBoxes)
            lbl_classes.append(np.array([b, b]))  # put one class per each box

        # return (train_images, {"boxes": lbl_boxes, "classes": lbl_classes})
        # retval = (
        #     np.array(train_images),
        #     # np.array(np.hstack((lbl_classes, lbl_boxes))),
        #     y,
        # )

        se = SamplesEncoder()
        retval = se.encode_batch(
            np.array(train_images), np.array(lbl_boxes), np.array(lbl_classes)
        )

        return retval
        # return np.array(train_images), np.array(np.hstack((lbl_classes, lbl_boxes)))

    def __data_generation(self, image_keys_temp):
        batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
        # batch_keypoints = np.empty(
        #     (self.batch_size, 1, 1, NUM_KEYPOINTS), dtype="float32"
        # )

        # for i, key in enumerate(image_keys_temp):
        #     data = [] # get_dog(key)
        #     current_keypoint = np.array(data["joints"])[:, :2]
        #     kps = []

        #     # To apply our data augmentation pipeline, we first need to
        #     # form Keypoint objects with the original coordinates.
        #     for j in range(0, len(current_keypoint)):
        #         kps.append(Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))

        #     # We then project the original image and its keypoint coordinates.
        #     current_image = data["img_data"]
        #     kps_obj = KeypointsOnImage(kps, shape=current_image.shape)

        #     # Apply the augmentation pipeline.
        #     (new_image, new_kps_obj) = self.aug(image=current_image, keypoints=kps_obj)
        #     batch_images[i,] = new_image

        #     # Parse the coordinates from the new keypoint object.
        #     kp_temp = []
        #     for keypoint in new_kps_obj:
        #         kp_temp.append(np.nan_to_num(keypoint.x))
        #         kp_temp.append(np.nan_to_num(keypoint.y))

        #     # More on why this reshaping later.
        #     batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 24 * 2)

        # # Scale the coordinates to [0, 1] range.
        # batch_keypoints = batch_keypoints / IMG_SIZE

        # return (batch_images, batch_keypoints)
        Image = []
        BoundingBoxes = []
        ClassLabels = []
        return (Image, BoundingBoxes, ClassLabels)
