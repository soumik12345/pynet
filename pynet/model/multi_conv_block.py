import tensorflow as tf
from tensorflow import keras


class MultiConvolutionBlock(keras.layers.Layer):
    def __init__(self, filters: int, max_conv_size: int, apply_norm: bool, **kwargs):
        super().__init__(**kwargs)
