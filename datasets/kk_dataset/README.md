TODO(kk_dataset): Markdown description of that will appear on the catalog page.
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):

Reference
https://www.tensorflow.org/datasets/add_dataset

Tutorial
cd /home/jiri/EfficientDet/datasets/kk_dataset
tfds build
cd ~/tensorflow_datasets/kk_dataset/1.0.1

# Create detector dataset from Studio
1) Setup geometry, attach aruco as ancor to device element.
2) Open Functions page, add localization.
3) Capture image to hard drive.
4) Open EfficientDet repo, cd to datasets/kk_dataset
5) Edit 'kk_dataset_dataset_builder.py', increase version, add notes, set source 'annotation.csv'
6) edit annotation file to match 'TRAINING,/mnt/c/..path,xmin,ymin,,,xmax,ymax,,'
6) tfds build