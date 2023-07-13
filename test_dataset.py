from dataset import MyDataset, CSVDataset, image_mosaic

DATA_PATH = "/mnt/d/dev/keypoints/rv12_dataset_v2"
BATCH_SIZE = 8

# ds = MyDataset(DATA_PATH, None, BATCH_SIZE)
ds = CSVDataset("/home/jiri/EfficientDet/meta_split.csv", None, BATCH_SIZE)
n = ds.__len__()

for i in range(0, 5):
    images, labels = ds.__getitem__(i)
    image_mosaic(images)
