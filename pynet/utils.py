import numpy as np


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
