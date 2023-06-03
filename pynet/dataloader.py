import os
from glob import glob

import tensorflow as tf

from wandb_addons.utils import fetch_wandb_artifact


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
