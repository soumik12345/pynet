import numpy as np
import tensorflow as tf


def preprocess_raw_image(raw_image):
    raw_combined = np.dstack(
        (
            raw_image[1::2, 1::2],
            raw_image[0::2, 1::2],
            raw_image[0::2, 0::2],
            raw_image[1::2, 0::2],
        )
    )
    return raw_combined.astype(np.float32) / (4 * 255)


def initialize_gpus():
    """Reference: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
