import keras
import numpy as np
import tensorflow as tf
from model.anchors import SamplesEncoder
import os
import json

IMG_FILE_SIZE = 320.0
IMG_OUT_SIZE = 256.0
IMG_SCALE = IMG_OUT_SIZE / IMG_FILE_SIZE


def get_image_names(data_dir):
    for root_dir, cur_dir, files in os.walk(data_dir):
        return [file for file in files if os.path.splitext(file)[1] == ".png"]


def is_box(val):
    return val["shape_type"] == "rectangle"


def is_box_point(val, groupid):
    return val["shape_type"] == "point" and val["group_id"] == groupid


def labelme_to_coco(labelme_data):
    shapes = labelme_data["shapes"]
    image_width = labelme_data["imageWidth"]
    image_height = labelme_data["imageHeight"]
    bboxes = [shape for shape in shapes if is_box(shape)]

    # out_classes = []  # group == class_id

    objs = []
    for bbox in bboxes:
        out_points = []
        group_id = bbox["group_id"]

        x1, y1 = bbox["points"][0]
        x2, y2 = bbox["points"][1]

        if x1 > x2:
            tmp = x2
            x2 = x1
            x1 = tmp

        if y1 > y2:
            tmp = y2
            y2 = y1
            y1 = tmp

        # int_box = [
        #     int(item) for item in [x1, y1, x2 - x1, y2 - y1]
        # ]  # convert to integers

        box_keypoints = [shape for shape in shapes if is_box_point(shape, group_id)]
        box_keypoints = sorted(box_keypoints, key=lambda x: x["label"])

        my_points = []
        for point in box_keypoints:
            my_point = [int(item) for item in point["points"][0]]
            my_point.append(1)
            my_points.append(my_point)

        out_points.append(my_points)

        objs.append(
            {
                "class_id": group_id - 1,
                "box": np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32),
                # "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                "keypoints": out_points,
            }
        )

    # coco_data = {"classes": out_classes, {"bboxes": out_boxes, "keypoints": out_points}}

    return objs


class MyDataset(keras.utils.Sequence):
    def __init__(self, data_dir, aug, batch_size, train=True):
        self.data_dir = data_dir
        self.aug = aug
        self.batch_size = batch_size
        self.train = train
        self.on_epoch_end()
        self.se = SamplesEncoder()

        # self.images_dir = self.data_dir + "/images"
        # self.annot_dir = self.data_dir + "/annotations"

        self.image_names = get_image_names(self.data_dir)

        self.cache = self.get_batch(0)

    def __len__(self):
        num_batches_in_dataset = int(
            (len(self.image_names) + self.batch_size - 1) / self.batch_size
        )
        return num_batches_in_dataset

    def __getitem__(self, index):
        return self.cache

    def get_batch(self, index):
        train_images = []
        lbl_boxes = []
        lbl_classes = []

        idx_from = index * self.batch_size
        idx_to = min(idx_from + self.batch_size, len(self.image_names))

        for idx in range(idx_from, idx_to):
            image_path = self.data_dir + "/" + self.image_names[idx]
            label_path = (
                self.data_dir
                + "/"
                + os.path.splitext(self.image_names[idx])[0]
                + ".json"
            )

            Image = tf.keras.utils.load_img(image_path)
            Image = Image.resize((256, 256))
            Image = tf.keras.utils.img_to_array(Image) / 255.0

            train_images.append(Image)

            # boxes
            with open(label_path, "r") as json_file:
                label = json.load(json_file)
                objs = labelme_to_coco(label)

                num_boxes = len(objs)

                BoundingBoxes = np.zeros((num_boxes, 4), dtype=np.float32)
                Classes = np.zeros((num_boxes,), dtype=np.float32)
                for iobj, obj in enumerate(objs):
                    BoundingBoxes[iobj] = obj["box"] * IMG_SCALE
                    Classes[iobj] = obj["class_id"]

                lbl_boxes.append(BoundingBoxes)
                lbl_classes.append(Classes)

        if (
            not (len(lbl_boxes) == len(lbl_classes) == len(train_images))
            or len(train_images) < 1
        ):
            raise ValueError(
                "Unexpected dimension of MyDataset. The size of boxes: {:d}, classes: {:d} and images: {:d}; batch index: {:d}, files from: {:d}, to: {:d}".format(
                    len(lbl_boxes),
                    len(lbl_classes),
                    len(train_images),
                    index,
                    idx_from,
                    idx_to,
                )
            )
        retval = self.se.encode_batch(np.array(train_images), lbl_boxes, lbl_classes)

        return retval

        #     src_boxes = label["boxes"]
        #     num_boxes = len(src_boxes)

        #     BoundingBoxes = np.zeros((num_boxes, 4))
        #     for ibox, box in enumerate(src_boxes):
        #         BoundingBoxes[ibox][0] = box[0]  # x
        #         BoundingBoxes[ibox][1] = box[1]  # y
        #         BoundingBoxes[ibox][2] = box[2] - box[0]  # width
        #         BoundingBoxes[ibox][3] = box[3] - box[1]  # height

        # lbl_boxes.append(BoundingBoxes)
        # # each box is of the format [x, y, width, height]
        # BoundingBoxes = np.array(
        #     [[111, 130, 133, 183], [189, 133, 214, 187]]
        # ).astype(
        #     np.single
        # )  # center, dimension

        # drazka = 0
        # nedrazka = 1

        # for b in range(0, self.batch_size):
        #     train_images.append(Image)
        #     lbl_boxes.append(BoundingBoxes)
        #     lbl_classes.append(
        #         np.array([drazka, nedrazka])
        #     )  # put one class per each box

        # retval = self.se.encode_batch(
        #     np.array(train_images), np.array(lbl_boxes), np.array(lbl_classes)
        # )

        # return retval
        # return np.array(train_images), np.array(np.hstack((lbl_classes, lbl_boxes)))
