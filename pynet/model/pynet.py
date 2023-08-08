import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras_core as keras
import math
from .modules import level_0, level_1, level_2, level_3, level_4, level_5
from .multi_conv_block import MultiConvolutionBlock
from pynet.losses import MultiScaleSSIMLoss, PerceptualLoss


class PyNet(keras.Model):
    def __init__(
        self,
        input_size: int = 224,
        apply_norm: bool = True,
        apply_norm_l1: bool = False,
        use_sigmoid: bool = True,
        return_lower_level_outputs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.network = self.build_network(
            input_size,
            apply_norm,
            apply_norm_l1,
            use_sigmoid,
            return_lower_level_outputs,
        )

        self.input_size = input_size
        self.apply_norm = apply_norm
        self.apply_norm_l1 = apply_norm_l1
        self.use_sigmoid = use_sigmoid
        self.return_lower_level_outputs = return_lower_level_outputs

    def build_network(
        self,
        input_size,
        apply_norm,
        apply_norm_l1,
        use_sigmoid,
        return_lower_level_outputs,
    ):
        inputs = keras.Input((input_size, input_size, 4))
        # First pass for calculating level 2,3,4,5 inputs
        # Level 1 first block out
        l1_out_1 = MultiConvolutionBlock(
            filters=32, max_conv_size=3, apply_norm=apply_norm_l1
        )(inputs)
        # Pool for Level 2 input
        l1_pool = keras.layers.MaxPool2D(pool_size=(2, 2))(l1_out_1)
        # Level 2 first block out
        l2_out_1 = MultiConvolutionBlock(
            filters=64, max_conv_size=3, apply_norm=apply_norm
        )(l1_pool)
        # Pool for Level 3 input
        l2_pool = keras.layers.MaxPool2D(pool_size=(2, 2))(l2_out_1)
        # Level 3 first block out
        l3_out_1 = MultiConvolutionBlock(
            filters=128, max_conv_size=3, apply_norm=apply_norm
        )(l2_pool)
        # Pool for Level 4 input
        l3_pool = keras.layers.MaxPool2D(pool_size=(2, 2))(l3_out_1)
        # Level 4 first block out
        l4_out_1 = MultiConvolutionBlock(
            filters=256, max_conv_size=3, apply_norm=apply_norm
        )(l3_pool)
        # Pool for Level 5 input
        l4_pool = keras.layers.MaxPool2D(pool_size=(2, 2))(l4_out_1)

        l5_out_final, l5_pass_a, l5_pass_b = level_5(
            l4_pool=l4_pool, apply_norm=apply_norm, use_sigmoid=use_sigmoid
        )  # 14x14x3 Final out shape
        l4_out_final, l4_pass_a, l4_pass_b = level_4(
            l4_out_1=l4_out_1,
            l5_pass_a=l5_pass_a,
            l5_pass_b=l5_pass_b,
            apply_norm=apply_norm,
            use_sigmoid=use_sigmoid,
        )  # 28x28x3 Final out shape
        l3_out_final, l3_pass_a, l3_pass_b = level_3(
            l3_out_1=l3_out_1,
            l4_pass_a=l4_pass_a,
            l4_pass_b=l4_pass_b,
            apply_norm=apply_norm,
            use_sigmoid=use_sigmoid,
        )  # 56x56x3 Final out shape
        l2_out_final, l2_pass_a, l2_pass_b = level_2(
            l2_out_1=l2_out_1,
            l3_pass_a=l3_pass_a,
            l3_pass_b=l3_pass_b,
            apply_norm=apply_norm,
            use_sigmoid=use_sigmoid,
        )  # 112x112x3 Final out shape
        l1_out_final, l1_pass = level_1(
            l1_out_1=l1_out_1,
            l2_pass_a=l2_pass_a,
            l2_pass_b=l2_pass_b,
            apply_norm=apply_norm_l1,
            use_sigmoid=use_sigmoid,
        )  # 224x224x3 Final out shape
        l0_out_final = level_0(l1_pass, use_sigmoid)  # 448x448x3 Final out shape

        outputs = (
            [
                l0_out_final,
                l1_out_final,
                l2_out_final,
                l3_out_final,
                l4_out_final,
                l5_out_final,
            ]
            if return_lower_level_outputs
            else l0_out_final
        )

        return keras.Model(inputs, outputs)

    def get_config(self):
        return {
            "input_size": self.input_size,
            "apply_norm": self.apply_norm,
            "apply_norm_l1": self.apply_norm_l1,
            "use_sigmoid": self.use_sigmoid,
            "return_lower_level_outputs": self.return_lower_level_outputs,
        }

    def call(self, inputs):
        return self.network(inputs)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.network.save_weights(filepath, overwrite, save_format, options)

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        self.network.load_weights(filepath, skip_mismatch, by_name, options)

    def compile(self, *args, **kwargs):
        self.mean_squared_error = keras.losses.MeanSquaredError()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = MultiScaleSSIMLoss()
        # Train Loss Trackers
        self.total_loss_train_tracker = keras.metrics.Mean(name="total_loss_train")
        self.l0_loss_train_tracker = keras.metrics.Mean(name="l0_loss_train")
        self.l1_loss_train_tracker = keras.metrics.Mean(name="l1_loss_train")
        self.l2_loss_train_tracker = keras.metrics.Mean(name="l2_loss_train")
        self.l3_loss_train_tracker = keras.metrics.Mean(name="l3_loss_train")
        self.l4_loss_train_tracker = keras.metrics.Mean(name="l4_loss_train")
        self.l5_loss_train_tracker = keras.metrics.Mean(name="l5_loss_train")
        # Test Loss Trackers
        # MSE Loss
        self.l0_mse_loss_tracker = keras.metrics.Mean(name="l0_mse_loss")
        self.l1_mse_loss_tracker = keras.metrics.Mean(name="l1_mse_loss")
        self.l2_mse_loss_tracker = keras.metrics.Mean(name="l2_mse_loss")
        self.l3_mse_loss_tracker = keras.metrics.Mean(name="l3_mse_loss")
        self.l4_mse_loss_tracker = keras.metrics.Mean(name="l4_mse_loss")
        self.l5_mse_loss_tracker = keras.metrics.Mean(name="l5_mse_loss")
        # PSNR Loss
        self.l0_psnr_loss_tracker = keras.metrics.Mean(name="l0_psnr_loss")
        self.l1_psnr_loss_tracker = keras.metrics.Mean(name="l1_psnr_loss")
        self.l2_psnr_loss_tracker = keras.metrics.Mean(name="l2_psnr_loss")
        self.l3_psnr_loss_tracker = keras.metrics.Mean(name="l3_psnr_loss")
        self.l4_psnr_loss_tracker = keras.metrics.Mean(name="l4_psnr_loss")
        self.l5_psnr_loss_tracker = keras.metrics.Mean(name="l5_psnr_loss")
        # SSIM Loss
        self.l0_ssim_tracker = keras.metrics.Mean(name="l0_ssim")
        self.l1_ssim_tracker = keras.metrics.Mean(name="l1_ssim")
        # Perceptual Loss
        self.l0_perceptual_loss_tracker = keras.metrics.Mean(name="l0_perceptual_loss")
        self.l1_perceptual_loss_tracker = keras.metrics.Mean(name="l1_perceptual_loss")
        self.l2_perceptual_loss_tracker = keras.metrics.Mean(name="l2_perceptual_loss")
        self.l3_perceptual_loss_tracker = keras.metrics.Mean(name="l3_perceptual_loss")
        self.l4_perceptual_loss_tracker = keras.metrics.Mean(name="l4_perceptual_loss")

        super().compile(*args, **kwargs)

    @property
    def train_metrics(self):
        return [
            self.total_loss_train_tracker,
            self.l0_loss_train_tracker,
            self.l1_loss_train_tracker,
            self.l2_loss_train_tracker,
            self.l3_loss_train_tracker,
            self.l4_loss_train_tracker,
            self.l5_loss_train_tracker,
        ]

    @property
    def eval_metrics(self):
        return [
            # MSE Loss
            self.l0_mse_loss_tracker,
            self.l1_mse_loss_tracker,
            self.l2_mse_loss_tracker,
            self.l3_mse_loss_tracker,
            self.l4_mse_loss_tracker,
            self.l5_mse_loss_tracker,
            # PSNR Loss
            self.l0_psnr_loss_tracker,
            self.l1_psnr_loss_tracker,
            self.l2_psnr_loss_tracker,
            self.l3_psnr_loss_tracker,
            self.l4_psnr_loss_tracker,
            self.l5_psnr_loss_tracker,
            # SSIM Loss
            self.l0_ssim_tracker,
            self.l1_ssim_tracker,
            # Perceptual Loss
            self.l0_perceptual_loss_tracker,
            self.l1_perceptual_loss_tracker,
            self.l2_perceptual_loss_tracker,
            self.l3_perceptual_loss_tracker,
            self.l4_perceptual_loss_tracker,
        ]

    def compute_train_losses(self, ground_truths, outputs):
        (
            l0_out_final,
            l1_out_final,
            l2_out_final,
            l3_out_final,
            l4_out_final,
            l5_out_final,
        ) = outputs
        (
            l0_ground_truth,
            l1_ground_truth,
            l2_ground_truth,
            l3_ground_truth,
            l4_ground_truth,
            l5_ground_truth,
        ) = ground_truths

        l5_loss = self.mean_squared_error(l5_ground_truth, l5_out_final)
        l4_loss = self.mean_squared_error(l4_ground_truth, l4_out_final)
        l3_loss = 10.0 * self.mean_squared_error(
            l3_ground_truth, l3_out_final
        ) + self.perceptual_loss(l3_ground_truth, l3_out_final)
        l2_loss = 10.0 * self.mean_squared_error(
            l2_ground_truth, l2_out_final
        ) + self.perceptual_loss(l2_ground_truth, l2_out_final)
        l1_loss = 10.0 * self.mean_squared_error(
            l1_ground_truth, l1_out_final
        ) + self.perceptual_loss(l1_ground_truth, l1_out_final)
        l0_loss = (
            self.mean_squared_error(l0_ground_truth, l0_out_final)
            + self.perceptual_loss(l0_ground_truth, l0_out_final)
            + 0.4 * (1 - self.ssim_loss(l0_ground_truth, l0_out_final))
        )
        total_loss = l0_loss + l1_loss + l2_loss + l3_loss + l4_loss + l5_loss
        return {
            "l0_loss_train": l0_loss,
            "l1_loss_train": l1_loss,
            "l2_loss_train": l2_loss,
            "l3_loss_train": l3_loss,
            "l4_loss_train": l4_loss,
            "l5_loss_train": l5_loss,
            "total_loss_train": total_loss,
        }

    def train_step(self, data):
        inputs, ground_truths = data
        with tf.GradientTape() as tape:
            outputs = self.network(inputs)
            losses = self.compute_train_losses(ground_truths, outputs)

        gradients = tape.gradient(
            losses["total_loss_train"], self.network.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.total_loss_train_tracker.update(losses["total_loss_train"])
        self.l0_loss_train_tracker.update(losses["l0_loss_train"])
        self.l1_loss_train_tracker.update(losses["l1_loss_train"])
        self.l2_loss_train_tracker.update(losses["l2_loss_train"])
        self.l3_loss_train_tracker.update(losses["l3_loss_train"])
        self.l4_loss_train_tracker.update(losses["l4_loss_train"])
        self.l5_loss_train_tracker.update(losses["l5_loss_train"])

        return {metric.name: metric.result() for metric in self.train_metrics}

    def compute_eval_losses(self, ground_truths, outputs):
        (
            l0_out_final,
            l1_out_final,
            l2_out_final,
            l3_out_final,
            l4_out_final,
            l5_out_final,
        ) = outputs
        (
            l0_ground_truth,
            l1_ground_truth,
            l2_ground_truth,
            l3_ground_truth,
            l4_ground_truth,
            l5_ground_truth,
        ) = ground_truths

        # MSE and PSNR for all levels
        l5_mse_loss = self.mean_squared_error(l5_ground_truth, l5_out_final)
        l4_mse_loss = self.mean_squared_error(l4_ground_truth, l4_out_final)
        l3_mse_loss = self.mean_squared_error(l3_ground_truth, l3_out_final)
        l2_mse_loss = self.mean_squared_error(l2_ground_truth, l2_out_final)
        l1_mse_loss = self.mean_squared_error(l1_ground_truth, l1_out_final)
        l0_mse_loss = self.mean_squared_error(l0_ground_truth, l0_out_final)

        l5_psnr_loss = tf.image.psnr(l5_ground_truth, l5_out_final)
        l4_psnr_loss = tf.image.psnr(l4_ground_truth, l4_out_final)
        l3_psnr_loss = tf.image.psnr(l3_ground_truth, l3_out_final)
        l2_psnr_loss = tf.image.psnr(l2_ground_truth, l2_out_final)
        l1_psnr_loss = tf.image.psnr(l1_ground_truth, l1_out_final)
        l0_psnr_loss = tf.image.psnr(l0_ground_truth, l0_out_final)

        # SSIM for Level 0 and Level 1
        l0_ssim = self.ssim_loss(l0_ground_truth, l0_out_final)
        l1_ssim = self.ssim_loss(l1_ground_truth, l1_out_final)

        # Perceptual loss for Level 0, 1, 2, 3, 4
        l0_perceptual_loss = self.perceptual_loss(l0_ground_truth, l0_out_final)
        l1_perceptual_loss = self.perceptual_loss(l1_ground_truth, l1_out_final)
        l2_perceptual_loss = self.perceptual_loss(l2_ground_truth, l2_out_final)
        l3_perceptual_loss = self.perceptual_loss(l3_ground_truth, l3_out_final)
        l4_perceptual_loss = self.perceptual_loss(l4_ground_truth, l4_out_final)

        return {
            # MSE Loss
            "l0_mse_loss": l0_mse_loss,
            "l1_mse_loss": l1_mse_loss,
            "l2_mse_loss": l2_mse_loss,
            "l3_mse_loss": l3_mse_loss,
            "l4_mse_loss": l4_mse_loss,
            "l5_mse_loss": l5_mse_loss,
            # PSNR Loss
            "l0_psnr_loss": l0_psnr_loss,
            "l1_psnr_loss": l1_psnr_loss,
            "l2_psnr_loss": l2_psnr_loss,
            "l3_psnr_loss": l3_psnr_loss,
            "l4_psnr_loss": l4_psnr_loss,
            "l5_psnr_loss": l5_psnr_loss,
            # SSIM Loss
            "l0_ssim": l0_ssim,
            "l1_ssim": l1_ssim,
            # Perceptual Loss
            "l0_perceptual_loss": l0_perceptual_loss,
            "l1_perceptual_loss": l1_perceptual_loss,
            "l2_perceptual_loss": l2_perceptual_loss,
            "l3_perceptual_loss": l3_perceptual_loss,
            "l4_perceptual_loss": l4_perceptual_loss,
        }

    def test_step(self, data):
        inputs, ground_truths = data
        outputs = self.network(inputs)
        losses = self.compute_eval_losses(ground_truths, outputs)

        # MSE Loss
        self.l0_mse_loss_tracker.update(losses["l0_mse_loss"])
        self.l1_mse_loss_tracker.update(losses["l1_mse_loss"])
        self.l2_mse_loss_tracker.update(losses["l2_mse_loss"])
        self.l3_mse_loss_tracker.update(losses["l3_mse_loss"])
        self.l4_mse_loss_tracker.update(losses["l4_mse_loss"])
        self.l5_mse_loss_tracker.update(losses["l5_mse_loss"])
        # PSNR Loss
        self.l0_psnr_loss_tracker.update(losses["l0_psnr_loss"])
        self.l1_psnr_loss_tracker.update(losses["l1_psnr_loss"])
        self.l2_psnr_loss_tracker.update(losses["l2_psnr_loss"])
        self.l3_psnr_loss_tracker.update(losses["l3_psnr_loss"])
        self.l4_psnr_loss_tracker.update(losses["l4_psnr_loss"])
        self.l5_psnr_loss_tracker.update(losses["l5_psnr_loss"])
        # SSIM Loss
        self.l0_ssim_tracker.update(losses["l0_ssim"])
        self.l1_ssim_tracker.update(losses["l1_ssim"])
        # Perceptual Loss
        self.l0_perceptual_loss_tracker.update(losses["l0_perceptual_loss"])
        self.l1_perceptual_loss_tracker.update(losses["l1_perceptual_loss"])
        self.l2_perceptual_loss_tracker.update(losses["l2_perceptual_loss"])
        self.l3_perceptual_loss_tracker.update(losses["l3_perceptual_loss"])
        self.l4_perceptual_loss_tracker.update(losses["l4_perceptual_loss"])

        return {metric.name: metric.result() for metric in self.eval_metrics}
