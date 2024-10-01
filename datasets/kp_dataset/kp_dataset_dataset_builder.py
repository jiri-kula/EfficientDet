"""kp_dataset dataset."""

import tensorflow_datasets as tfds
import keypoint_extractor

IMAGE_SIZE = 224

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for kp_dataset dataset."""

    VERSION = tfds.core.Version('1.0.11')
    RELEASE_NOTES = {
        '1.0.11': 'detector_dataset_11 - extensive rotations',
        '1.0.1': 'detector_dataset_9 - JPEG 30% quality',
        '1.0.0': 'detector_dataset_13',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(kp_dataset): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
                    "objects": tfds.features.Sequence(
                        {
                            "bbox": tfds.features.BBoxFeature(),
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
                path="/mnt/c/local/tmp/detector_dataset_11/annotation.csv"
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(kk_dataset): Yields (key, example) tuples from the dataset
        for image_path, image, objects in keypoint_extractor.read_csv_file(path):
            yield image_path, {
                "image": image,
                "objects": objects,
            }
