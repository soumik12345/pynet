import tensorflow as tf
from tensorflow import keras


class SSIMLoss(keras.losses.Loss):
    def __init__(
        self,
        max_val: float = 1.0,
        reduction=keras.utils.losses_utils.ReductionV2.AUTO,
        name=None,
    ):
        self.max_val = max_val
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=self.max_val))


class MultiScaleSSIMLoss(keras.losses.Loss):
    def __init__(
        self,
        max_val: float = 1.0,
        reduction=keras.utils.losses_utils.ReductionV2.AUTO,
        name=None,
    ):
        self.max_val = max_val
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.image.ssim_multiscale(y_true, y_pred, max_val=self.max_val)
        )
