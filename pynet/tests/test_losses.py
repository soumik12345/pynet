import unittest

import tensorflow as tf

from pynet.losses import MultiScaleSSIMLoss, PerceptualLoss, SSIMLoss


class LossTester(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.ssim_loss = SSIMLoss()
        self.multi_scale_ssim_loss = MultiScaleSSIMLoss()
        self.perceptual_loss = PerceptualLoss(rescale_inputs=False)
        self.sample_y = tf.ones((1, 224, 224, 3))

    def test_ssim_loss(self):
        self.assertEqual(self.ssim_loss(self.sample_y, self.sample_y).numpy(), 1.0)

    def test_multi_scale_ssim_loss(self):
        self.assertEqual(
            self.multi_scale_ssim_loss(self.sample_y, self.sample_y).numpy(), 1.0
        )

    def test_perceptual_loss(self):
        self.assertEqual(
            self.perceptual_loss(self.sample_y, self.sample_y).numpy(), 0.0
        )
