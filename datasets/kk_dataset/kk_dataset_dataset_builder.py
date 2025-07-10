"""kk_dataset dataset."""

import tensorflow_datasets as tfds
import adaptive_strategy

# relevat datasets: 8, 9, 10, 11, 12, 13
# dataset 14: tripos 2/3 near detail 1280, random position

# versions:
# 1.0.x: initial release
# 1.1.x: remake of 1.0.x with random roi around each object - 1 sample with all object, 1 negative image + 5 random roi, so that each full frame image generates 7 training images

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for kk_dataset dataset."""

    VERSION = tfds.core.Version("1.4.25")
    RELEASE_NOTES = {
        "1.4.25": "Contains HD datasets 25, no validation, morning sunlight, overexposed.",
        "1.4.23": "Contains HD datasets 23 as train, 24 as validation.",
        "1.3.22": "Contains datasets 21 as train, 22 as validation.",
        "1.3.21": "Contains datasets 21 as train, 20 as validation, 17 as test.",
        "1.2.3": "Contains datasets 17. Arch 1, variable ETime, bbox fits",
        "1.2.2": "DO NOT USE: Contains datasets 16. Arch 1, variable ETime, bbox fits",
        "1.2.1": "DO NOT USE: Contains datasets 15. Arch 1, constant ETime",
        "1.1.20": "Contains datasets 20.",
        "1.1.11": "Contains datasets 13.",
        "1.1.11": "Contains datasets 12.",
        "1.1.11": "Contains datasets 11.",
        "1.1.10": "Contains datasets 10.",
        "1.1.9": "Contains datasets 9.",
        "1.1.8": "Contains datasets 8.",
        "1.0.15": "Contains datasets 14. tripos 2/3 near detail 1280, random position.",
        "1.0.14": "Contains datasets 14. tripos 2/3 near detail 1280.",
        "1.0.11": "Remake of 1.0.2 with random roi around each object.",
        "1.0.4": "Contains datasets 13. Capture static position at cloudy morning, ET 200-1000us.",
        "1.0.3": "Contains datasets 12. Capture static position at night, ET 400-3000us.",
        "1.0.2": "Contains datasets 11.",
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
                            "label": tfds.features.ClassLabel(names=["roh", "lb", "rb", "lt"]),
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
                path="/mnt/c/local/tmp/detector_dataset_25_hd/annotation.csv"
            )
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(kk_dataset): Yields (key, example) tuples from the dataset
        for image_path, image, objects in adaptive_strategy.read_csv_file(path):
            yield image_path, {
                "image": image,
                "objects": objects,
            }
