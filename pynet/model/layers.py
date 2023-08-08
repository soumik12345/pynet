import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras_core as keras


class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, padding: int, **kwargs):
        self.padding = padding
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config

    def call(self, inputs):
        return tf.pad(
            inputs,
            [
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
            "REFLECT",
        )


class UpSampleConvLayer(keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        upsample_factor: int = 2,
        strides: int = 1,
        apply_activation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample_factor = upsample_factor
        self.strides = strides
        self.apply_activation = apply_activation

        self.upsample = keras.layers.UpSampling2D(
            size=upsample_factor, interpolation="bilinear"
        )
        self.reflection_padding = ReflectionPadding2D(padding=kernel_size // 2)
        self.convolution = keras.layers.Conv2D(filters, kernel_size, strides)
        self.activation = (
            keras.layers.LeakyReLU(alpha=0.2) if apply_activation else None
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "upsample_factor": self.upsample_factor,
                "strides": self.strides,
                "apply_activation": self.apply_activation,
            }
        )
        return config

    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.reflection_padding(x)
        x = self.convolution(x)
        return self.activation(x) if self.activation is not None else x


class ConvLayer(keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        upsample_factor: int = 2,
        strides: int = 1,
        apply_activation: bool = True,
        apply_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample_factor = upsample_factor
        self.strides = strides
        self.apply_activation = apply_activation
        self.apply_norm = apply_norm

        self.reflection_padding = ReflectionPadding2D(padding=kernel_size // 2)
        self.convolution = keras.layers.Conv2D(filters, kernel_size, strides)
        self.activation = (
            keras.layers.LeakyReLU(alpha=0.2) if apply_activation else None
        )
        self.norm = (
            keras.layers.GroupNormalization(groups=filters) if self.apply_norm else None
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "upsample_factor": self.upsample_factor,
                "strides": self.strides,
                "apply_activation": self.apply_activation,
                "apply_norm": self.apply_norm,
            }
        )
        return config

    def call(self, inputs):
        x = self.reflection_padding(inputs)
        x = self.convolution(x)
        x = self.norm(x) if self.norm is not None else x
        return self.activation(x) if self.activation is not None else x
