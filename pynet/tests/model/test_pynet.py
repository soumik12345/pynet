import unittest
import tensorflow as tf
from pynet.model.modules import level_5, level_4, level_3, level_2, level_1, level_0
from pynet.model.pynet import PyNet
 
class ModelTester(unittest.TestCase):
    def test_level_5(self) -> None:
        use_sigmoid = True
        l4_pool = tf.ones((1, 14, 14, 256))
        l5_out_final, l5_pass_a, l5_pass_b = level_5(l4_pool,True,use_sigmoid)
        self.assertEqual(l5_out_final.shape,(1,14,14,3))
        self.assertEqual(l5_pass_a.shape,(1,28,28,256))
        self.assertEqual(l5_pass_b.shape,(1,28,28,256))
    
    def test_level_4(self) -> None:
        use_sigmoid = True
        l4_out_1 = tf.ones((1,28,28,256))
        l5_pass_a = tf.ones((1,28,28,256))
        l5_pass_b = tf.ones((1,28,28,256))
        l4_out_final,l4_pass_a,l4_pass_b = level_4(l4_out_1,l5_pass_a,l5_pass_b,True,use_sigmoid)
        self.assertEqual(l4_out_final.shape,(1,28,28,3))
        self.assertEqual(l4_pass_a.shape,(1,56,56,128))
        self.assertEqual(l4_pass_b.shape,(1,56,56,128))

    def test_level_3(self) -> None:
        use_sigmoid = True
        l3_out_1 = tf.ones((1,56,56,128))
        l4_pass_a = tf.ones((1,56,56,128))
        l4_pass_b = tf.ones((1,56,56,128))
        l3_out_final,l3_pass_a,l3_pass_b = level_3(l3_out_1,l4_pass_a,l4_pass_b,True,use_sigmoid)
        self.assertEqual(l3_out_final.shape,(1,56,56,3))
        self.assertEqual(l3_pass_a.shape,(1,112,112,64))
        self.assertEqual(l3_pass_b.shape,(1,112,112,64))

    def test_level_2(self) -> None:
        use_sigmoid = False
        l2_out_1 = tf.ones((1,112,112,64))
        l3_pass_a = tf.ones((1,112,112,64))
        l3_pass_b = tf.ones((1,112,112,64))
        l2_out_final,l2_pass_a,l2_pass_b = level_2(l2_out_1,l3_pass_a,l3_pass_b,True,use_sigmoid)
        self.assertEqual(l2_out_final.shape,(1,112,112,3))
        self.assertEqual(l2_pass_a.shape,(1,224,224,32))
        self.assertEqual(l2_pass_b.shape,(1,224,224,32))

    def test_level_1(self) -> None:
        use_sigmoid = True
        l1_out_1 = tf.ones((1,224,224,32))
        l2_pass_a = tf.ones((1,224,224,32))
        l2_pass_b = tf.ones((1,224,224,32))
        l1_out_final,l1_pass = level_1(l1_out_1,l2_pass_a,l2_pass_b,True,use_sigmoid)
        self.assertEqual(l1_out_final.shape,(1,224,224,3))
        self.assertEqual(l1_pass.shape,(1,448,448,16))
    
    def test_level_0(self) -> None:
        use_sigmoid = True
        l1_pass = tf.ones((1,448,448,16))
        l0_out_final = level_0(l1_pass,use_sigmoid)
        self.assertEqual(l0_out_final.shape,(1,448,448,3))
        

    def test_pynet(self) -> None:
        input_tensor = tf.ones((1,224,224,4))
        model = PyNet(apply_norm=True, apply_norm_l1=False, use_sigmoid=True, return_lower_level_outputs=True)
        l0_out_final, l1_out_final, l2_out_final, l3_out_final, l4_out_final, l5_out_final = model(input_tensor)
        self.assertEqual(l0_out_final.shape,(1,448,448,3))
        self.assertEqual(l1_out_final.shape,(1,224,224,3))
        self.assertEqual(l2_out_final.shape,(1,112,112,3))
        self.assertEqual(l3_out_final.shape,(1,56,56,3))
        self.assertEqual(l4_out_final.shape,(1,28,28,3))
        self.assertEqual(l5_out_final.shape,(1,14,14,3))

    def test_pynet_single_output(self) -> None:
        input_tensor = tf.ones((1,224,224,4))
        model = PyNet(apply_norm=True, apply_norm_l1=False, use_sigmoid=True, return_lower_level_outputs=False)
        l0_out_final = model(input_tensor)
        self.assertEqual(l0_out_final.shape,(1,448,448,3))


if __name__ == '__main__':
    unittest.main()
