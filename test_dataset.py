import dataset

DATA_PATH = "/mnt/d/dev/keypoints/rv12_dataset_v2"
BATCH_SIZE = 4

ds = dataset.MyDataset(DATA_PATH, None, BATCH_SIZE)
n = ds.__len__()

batch = ds.__getitem__(0)
batch = ds.__getitem__(int(n / BATCH_SIZE))
