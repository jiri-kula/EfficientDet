"""kk_dataset dataset."""

import tensorflow_datasets as tfds
import adaptive_strategy


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for kk_dataset dataset."""

    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.1": "Contains datasets 8, 9, 10.",
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(kk_dataset): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(320, 320, 3)),
                    "objects": tfds.features.Sequence(
                        {
                            "bbox": tfds.features.BBoxFeature(),
                            "label": tfds.features.ClassLabel(names=["roh", "vyrez"]),
                        }
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "objects"),  # Set to `None` to disable
            homepage="https://nullspaces.com",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(kk_dataset): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract("https://todo-data-url")

        # TODO(kk_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(
                path="/home/jiri/detector_datasets/8_9_10.csv"
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(kk_dataset): Yields (key, example) tuples from the dataset
        for image_path, image, objects in adaptive_strategy.read_csv_file(path):
            yield image_path, {
                "image": image,
                "objects": objects,
            }
