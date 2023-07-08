import os
from glob import glob

import tensorflow as tf

from .utils import fetch_wandb_artifact, preprocess_raw_image

_AUTOTUNE = tf.data.AUTOTUNE


class ZurichDatasetFactory:
    def __init__(self, artifact_address: str, val_split: float = 0.2) -> None:
        self.artifact_address = artifact_address
        self.val_split = val_split
        self.dataset_paths = self.fetch_dataset()

    def fetch_dataset(self):
        artifact_dir = fetch_wandb_artifact(
            artifact_address=self.artifact_address, artifact_type="dataset"
        )

        train_val_raw_images = sorted(
            glob(os.path.join(artifact_dir, "train", "huawei_raw", "*"))
        )
        train_val_dslr_images = sorted(
            glob(os.path.join(artifact_dir, "train", "canon", "*"))
        )

        num_train_images = len(train_val_dslr_images) - int(
            len(train_val_dslr_images) * self.val_split
        )

        train_raw_images = train_val_raw_images[:num_train_images]
        train_dslr_images = train_val_dslr_images[:num_train_images]

        val_raw_images = train_val_raw_images[num_train_images:]
        val_dslr_images = train_val_dslr_images[num_train_images:]

        test_raw_images = sorted(
            glob(os.path.join(artifact_dir, "test", "huawei_raw", "*"))
        )
        test_dslr_images = sorted(
            glob(os.path.join(artifact_dir, "test", "canon", "*"))
        )

        return {
            "Train": (train_raw_images, train_dslr_images),
            "Validation": (val_raw_images, val_dslr_images),
            "Test": (test_raw_images, test_dslr_images),
        }

    def preprocess_images(self, raw_image_file, dslr_image_file):
        raw_image = tf.cast(
            tf.image.decode_image(tf.io.read_file(raw_image_file)), dtype=tf.float32
        )
        dslr_image = tf.cast(
            tf.image.decode_image(tf.io.read_file(dslr_image_file)), dtype=tf.float32
        )

        def _preprocess(_raw_image, _dslsr_image):
            _raw_image = preprocess_raw_image(_raw_image.numpy())
            _dslsr_image = _dslsr_image.numpy() / 255.0
            return _raw_image, _dslsr_image

        return tf.py_function(
            _preprocess, [raw_image, dslr_image], [tf.float32, tf.float32]
        )

    def _build_dataset(self, dataset_paths, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(dataset_paths)
        dataset = dataset.map(self.preprocess_images, num_parallel_calls=_AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset.prefetch(_AUTOTUNE)

    def get_datasets(self, batch_size: int):
        datasets = {}
        for alias, paths in self.dataset_paths.items():
            datasets[alias] = self._build_dataset(paths, batch_size)
        return datasets
