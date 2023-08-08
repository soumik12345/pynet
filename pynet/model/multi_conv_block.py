import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras_core as keras

from .layers import ConvLayer


class MultiConvolutionBlock(keras.layers.Layer):
    def __init__(self, filters: int, max_conv_size: int, apply_norm: bool, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.max_conv_size = max_conv_size
        self.apply_norm = apply_norm

        self.conv_3a = ConvLayer(filters=filters, kernel_size=3, apply_norm=apply_norm)
        self.conv_3b = ConvLayer(filters=filters, kernel_size=3, apply_norm=apply_norm)

        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(
                filters=filters, kernel_size=5, apply_norm=apply_norm
            )
            self.conv_5b = ConvLayer(
                filters=filters, kernel_size=5, apply_norm=apply_norm
            )

        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(
                filters=filters, kernel_size=7, apply_norm=apply_norm
            )
            self.conv_7b = ConvLayer(
                filters=filters, kernel_size=7, apply_norm=apply_norm
            )

        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(
                filters=filters, kernel_size=9, apply_norm=apply_norm
            )
            self.conv_9b = ConvLayer(
                filters=filters, kernel_size=9, apply_norm=apply_norm
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "max_conv_size": self.max_conv_size,
                "apply_norm": self.apply_norm,
            }
        )
        return config

    def call(self, inputs):
        outputs = self.conv_3b(self.conv_3a(inputs))

        if self.max_conv_size >= 5:
            out_5 = self.conv_5b(self.conv_5a(inputs))
            outputs = keras.layers.Concatenate(axis=-1)([outputs, out_5])

        if self.max_conv_size >= 7:
            out_7 = self.conv_7b(self.conv_7a(inputs))
            outputs = keras.layers.Concatenate(axis=-1)([outputs, out_7])

        if self.max_conv_size >= 9:
            out_9 = self.conv_9b(self.conv_9a(inputs))
            outputs = keras.layers.Concatenate(axis=-1)([outputs, out_9])

        return outputs
