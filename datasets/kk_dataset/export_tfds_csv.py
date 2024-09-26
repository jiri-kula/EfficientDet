# %%
import os
import csv
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


# %%
no_box = 1 / 320.0
no_class = 0


def export_dataset(dataset_name, split="train", output_dir="output"):
    # Load the dataset
    ds = tfds.load(dataset_name, split=split, shuffle_files=False)

    # Create directories for images and annotations
    images_dir = os.path.join(output_dir, "images")
    annotations_file = os.path.join(output_dir, "annotations.csv")
    os.makedirs(images_dir, exist_ok=True)

    # Initialize a list to store annotation data
    annotations = []

    # Iterate over the dataset
    for i, example in enumerate(tqdm(ds.take(5), desc="Processing dataset")):
        # Extract image and convert to numpy array
        image = example["image"].numpy()
        image_path = os.path.join(images_dir, f"image_{i}.png")

        # Save image to file
        tf.keras.preprocessing.image.save_img(image_path, image)

        # Extract bounding boxes and class labels
        objects = example["objects"]
        bboxes = objects["bbox"]
        class_labels = objects["label"]

        if bboxes.shape[0] == 0:
            annotations.append([image_path, no_box, no_box, no_box, no_box, no_class])
        else:
            for bbox, label in zip(bboxes, class_labels):
                ymin, xmin, ymax, xmax = bbox.numpy()
                class_label = label.numpy()
                annotations.append([image_path, xmin, xmax, ymin, ymax, class_label])

    # Write annotations to CSV file
    with open(annotations_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "xmin", "xmax", "ymin", "ymax", "class_label"])
        writer.writerows(annotations)

    print(f"Dataset exported to {output_dir}")


# Example usage
export_dataset("kk_dataset:1.0.1")

# %%
