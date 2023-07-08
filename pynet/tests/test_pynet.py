import unittest

import tensorflow as tf

from pynet.model.modules import level_0, level_1, level_2, level_3, level_4, level_5
from pynet.model import PyNet


class ModelTester(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.l4_pool = tf.ones((1, 14, 14, 256))
        self.level_4_input = tf.ones((1, 28, 28, 256))
        self.level_3_input = tf.ones((1, 56, 56, 128))
        self.level_2_input = tf.ones((1, 112, 112, 64))
        self.level_1_input = tf.ones((1, 224, 224, 32))
        self.level_0_input = tf.ones((1, 448, 448, 16))
        self.pynet_input = tf.ones((1, 224, 224, 4))

    def test_level_5(self) -> None:
        use_sigmoid = True
        l5_out_final, l5_pass_a, l5_pass_b = level_5(self.l4_pool, True, use_sigmoid)
        self.assertEqual(l5_out_final.shape, (1, 14, 14, 3))
        self.assertEqual(l5_pass_a.shape, (1, 28, 28, 256))
        self.assertEqual(l5_pass_b.shape, (1, 28, 28, 256))

    def test_level_4(self) -> None:
        use_sigmoid = True
        l4_out_final, l4_pass_a, l4_pass_b = level_4(
            self.level_4_input,
            self.level_4_input,
            self.level_4_input,
            True,
            use_sigmoid,
        )
        self.assertEqual(l4_out_final.shape, (1, 28, 28, 3))
        self.assertEqual(l4_pass_a.shape, (1, 56, 56, 128))
        self.assertEqual(l4_pass_b.shape, (1, 56, 56, 128))

    def test_level_3(self) -> None:
        use_sigmoid = True
        l3_out_final, l3_pass_a, l3_pass_b = level_3(
            self.level_3_input,
            self.level_3_input,
            self.level_3_input,
            True,
            use_sigmoid,
        )
        self.assertEqual(l3_out_final.shape, (1, 56, 56, 3))
        self.assertEqual(l3_pass_a.shape, (1, 112, 112, 64))
        self.assertEqual(l3_pass_b.shape, (1, 112, 112, 64))

    def test_level_2(self) -> None:
        use_sigmoid = False
        l2_out_final, l2_pass_a, l2_pass_b = level_2(
            self.level_2_input,
            self.level_2_input,
            self.level_2_input,
            True,
            use_sigmoid,
        )
        self.assertEqual(l2_out_final.shape, (1, 112, 112, 3))
        self.assertEqual(l2_pass_a.shape, (1, 224, 224, 32))
        self.assertEqual(l2_pass_b.shape, (1, 224, 224, 32))

    def test_level_1(self) -> None:
        use_sigmoid = True
        l1_out_final, l1_pass = level_1(
            self.level_1_input,
            self.level_1_input,
            self.level_1_input,
            True,
            use_sigmoid,
        )
        self.assertEqual(l1_out_final.shape, (1, 224, 224, 3))
        self.assertEqual(l1_pass.shape, (1, 448, 448, 16))

    def test_level_0(self) -> None:
        use_sigmoid = True
        l0_out_final = level_0(self.level_0_input, use_sigmoid)
        self.assertEqual(l0_out_final.shape, (1, 448, 448, 3))

    def test_pynet(self) -> None:
        model = PyNet(
            apply_norm=True,
            apply_norm_l1=False,
            use_sigmoid=True,
            return_lower_level_outputs=True,
        )
        (
            l0_out_final,
            l1_out_final,
            l2_out_final,
            l3_out_final,
            l4_out_final,
            l5_out_final,
        ) = model(self.pynet_input)
        self.assertEqual(l0_out_final.shape, (1, 448, 448, 3))
        self.assertEqual(l1_out_final.shape, (1, 224, 224, 3))
        self.assertEqual(l2_out_final.shape, (1, 112, 112, 3))
        self.assertEqual(l3_out_final.shape, (1, 56, 56, 3))
        self.assertEqual(l4_out_final.shape, (1, 28, 28, 3))
        self.assertEqual(l5_out_final.shape, (1, 14, 14, 3))

    def test_pynet_single_output(self) -> None:
        model = PyNet(
            apply_norm=True,
            apply_norm_l1=False,
            use_sigmoid=True,
            return_lower_level_outputs=False,
        )
        l0_out_final = model(self.pynet_input)
        self.assertEqual(l0_out_final.shape, (1, 448, 448, 3))
