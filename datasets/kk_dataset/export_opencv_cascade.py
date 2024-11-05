# Filename: default_strategy.py
# Created Date: 2024-10-01
# Modified Date: 2024-10-01
# Author: Jiri Kula
# Description: This creates positive file 'pos.txt' for Opencv cascade training.

#%%
import csv

WIDTH = 1280
HEIGHT = 720

def store(clsnames, xmins, ymins, xmaxs, ymaxs, src_filepath):
    # process individual bounding boxes
    objects = [src_filepath, len(xmins)]
    for clsname, xmin, ymin, xmax, ymax in zip(clsnames, xmins, ymins, xmaxs, ymaxs):
        objects.append(xmin * WIDTH)
        objects.append(ymin * HEIGHT)
        objects.append((xmax - xmin) * WIDTH)
        objects.append((ymax - ymin) * HEIGHT)

    return objects


# import csv file and perform function on each row
def read_csv_file(source_ann_filepath):
    src_lastfilepath = None

    # center of adaptive bounding box
    purposes = []
    clsnames = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    with open(source_ann_filepath, "r") as file:
        reader = csv.reader(file)

        # last file path is None
        # read coordinates and file until file path is different or there are no more rows
        # if have coordinate - save them
        for row in reader:
            filepath = row[1]

            # applies for the first row of input file
            if src_lastfilepath is None:
                src_lastfilepath = filepath

            if filepath != src_lastfilepath:
                yield store(
                    clsnames, xmins, ymins, xmaxs, ymaxs, src_lastfilepath
                )

                # update
                src_lastfilepath = filepath

                # cleanup
                purposes = []
                clsnames = []
                xmins = []
                ymins = []
                xmaxs = []
                ymaxs = []

            # store this row
            purpose = row[0]
            clsname = row[2]
            xmin = float(row[3])
            ymin = float(row[4])
            xmax = float(row[7])
            ymax = float(row[8])

            purposes.append(purpose)
            clsnames.append(clsname)
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)

        # store last after all src rows were processed
        yield store(
            clsnames, xmins, ymins, xmaxs, ymaxs, src_lastfilepath
        )
#%%
import matplotlib.pyplot as plt
from matplotlib import patches

# function to display image with bounding boxes
def display_image(row):
    image_path = row[0]
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    num_object = row[1]
    for iobj in range(num_object):
        xmin = row[2 + iobj * 4]
        ymin = row[3 + iobj * 4]
        width = row[4 + iobj * 4]
        height = row[5 + iobj * 4]
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    # test the function
    i = 0
    for row in read_csv_file(
        "/mnt/c/local/tmp/detector_dataset_14/annotation.csv"
    ):
        # pass
        display_image(row)
        plt.show()
        i += 1
        if i > 10:
            break
    print("Done")
# %%
