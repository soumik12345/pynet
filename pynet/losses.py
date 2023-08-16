import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras_core as keras


class SSIMLoss(keras.losses.Loss):
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        self.max_val = max_val
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=self.max_val))


class MultiScaleSSIMLoss(keras.losses.Loss):
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        self.max_val = max_val
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.image.ssim_multiscale(y_true, y_pred, max_val=self.max_val)
        )


class PerceptualLoss(keras.losses.Loss):
    def __init__(self, rescale_inputs: bool, *args, **kwargs):
        """Reference: https://github.com/srihari-humbarwadi/srgan_tensorflow/blob/master/losses.py#L4"""
        super().__init__(*args, **kwargs)
        self.rescale_inputs = rescale_inputs
        self.mean_squared_error = keras.losses.MeanSquaredError(reduction="none")
        vgg = keras.applications.VGG19(include_top=False)
        vgg.trainable = False
        # Getting rid of the final pooling layer of VGG19
        self.vgg_feature_model = keras.Model(
            vgg.input, vgg.get_layer("block5_conv4").output
        )

    def call(self, y_true, y_pred):
        if self.rescale_inputs:
            y_true = y_true * 255.0
            y_pred = y_pred * 255.0
        y_true_features = self.vgg_feature_model(
            keras.applications.vgg19.preprocess_input(y_true)
        )
        y_pred_features = self.vgg_feature_model(
            keras.applications.vgg19.preprocess_input(y_pred)
        )
        return self.mean_squared_error(y_true_features, y_pred_features)
