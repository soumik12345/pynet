import tensorflow as tf
from tensorflow import keras


class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, padding: int, **kwargs):
        self.padding = padding
        super(ReflectionPadding2D, self).__init__(**kwargs)

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
        self.convolution = keras.layers.Conv2D(
            filters, kernel_size, strides, padding="same"
        )
        self.activation = keras.layers.ReLU() if apply_activation else None

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

    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.reflection_padding(x)
        x = self.convolution(x)
        return self.activation(x) if self.activation is not None else x
