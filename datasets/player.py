# %% loads tfds dataset and shows the bounding with class id over the image

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_and_display_dataset(dataset_name):
    # Load the dataset
    dataset, info = tfds.load(dataset_name, with_info=True, split="train")

    # Get the class names
    # class_names = info.features["labels"].names

    # Iterate over the dataset
    for idx, example in enumerate(tfds.as_numpy(dataset)):
        image = example["image"]
        bboxes = example["objects"]["bbox"]
        labels = example["objects"]["label"]

        # Display the image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Add bounding boxes
        for bbox, label in zip(bboxes, labels):
            ymin, xmin, ymax, xmax = bbox
            rect = patches.Rectangle(
                (xmin * image.shape[1], ymin * image.shape[0]),
                (xmax - xmin) * image.shape[1],
                (ymax - ymin) * image.shape[0],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            plt.text(
                xmin * image.shape[1],
                ymin * image.shape[0],
                label,
                bbox=dict(facecolor="yellow", alpha=0.5),
            )

        # plt.show()
        plt.savefig(f"agg/output_{idx}.png")
        plt.close(fig)


# Example usage
load_and_display_dataset("kk_dataset:1.3.21")
