import tensorflow as tf
from tensorflow import keras
from .modules import level_5, level_4, level_3, level_2, level_1, level_0
from .multi_conv_block import MultiConvolutionBlock

def PyNet(apply_norm=True, apply_norm_l1=False, use_sigmoid=True):

    input_tensor = tf.keras.Input((224,224,4))
    #First pass for calculating level 2,3,4,5 inputs
    #Level 1 first block out
    l1_out_1 = MultiConvolutionBlock(32, 3, apply_norm_l1)(input_tensor)
    #Pool for Level 2 input
    l1_pool = keras.layers.MaxPool2D((2,2))(l1_out_1)
    #Level 2 first block out
    l2_out_1 = MultiConvolutionBlock(64, 3, apply_norm)(l1_pool)
    #Pool for Level 3 input
    l2_pool = keras.layers.MaxPool2D((2,2))(l2_out_1)
    #Level 3 first block out
    l3_out_1 = MultiConvolutionBlock(128, 3, apply_norm)(l2_pool)
    #Pool for Level 4 input
    l3_pool = keras.layers.MaxPool2D((2,2))(l3_out_1)
    #Level 4 first block out
    l4_out_1 = MultiConvolutionBlock(256, 3, apply_norm)(l3_pool)
    #Pool for Level 5 input
    l4_pool = keras.layers.MaxPool2D((2,2))(l4_out_1)

    l5_out_final,l5_pass_a,l5_pass_b = level_5(l4_pool,apply_norm, use_sigmoid) # 14x14
    l4_out_final,l4_pass_a,l4_pass_b = level_4(l4_out_1, l5_pass_a, l5_pass_b, apply_norm, use_sigmoid) # 28x28
    l3_out_final,l3_pass_a,l3_pass_b = level_3(l3_out_1, l4_pass_a, l4_pass_b, apply_norm, use_sigmoid) # 56x56
    l2_out_final,l2_pass_a,l2_pass_b = level_2(l2_out_1, l3_pass_a, l3_pass_b, apply_norm, use_sigmoid) # 112x112
    l1_out_final,l1_pass = level_1(l1_out_1, l2_pass_a, l2_pass_b, apply_norm_l1, use_sigmoid) # 224x224
    l0_out_final = level_0(l1_pass, use_sigmoid) # 448x448
    
    return keras.Model(inputs=input_tensor, outputs=[l0_out_final,l1_out_final,l2_out_final,l3_out_final,l4_out_final,l5_out_final], name='PyNet')