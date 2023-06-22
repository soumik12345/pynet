import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from .layers import ConvLayer, UpSampleConvLayer
from .multi_conv_block import MultiConvolutionBlock

def level_5(l4_pool,apply_norm, use_sigmoid):
    # Expected Output shape for image of 224x224 -> 224/2**(4) -> 14x14x3 
    # Input -> 14x14x256
    l5_out_1 = MultiConvolutionBlock(512, 3, apply_norm)(l4_pool)
    l5_out_2 = MultiConvolutionBlock(512, 3, apply_norm)(l5_out_1)
    l5_out_3 = MultiConvolutionBlock(512, 3, apply_norm)(l5_out_2)
    l5_out_4 = MultiConvolutionBlock(512, 3, apply_norm)(l5_out_3)
    # Upsample for pass
    l5_pass_a = UpSampleConvLayer(256, 3)(l5_out_4)
    l5_pass_b = UpSampleConvLayer(256, 3)(l5_out_4)
    # Final Output
    l5_out_final = ConvLayer(3, kernel_size=3, strides=1, apply_activation=False)(l5_out_4)
    if use_sigmoid == True:
        l5_out_final = keras.activations.sigmoid(l5_out_final)
    else:
        # Extra Values from original TF implementation https://github.com/aiff22/PyNET/blob/16eeb71d94fb0c6ed40a5403ca5efd762974b2bf/model.py#LL40C8-L40C8
        l5_out_final = keras.activations.tanh(l5_out_final) * 0.58 + 0.5
    return l5_out_final,l5_pass_a,l5_pass_b

def level_4(l4_out_1, l5_pass_a, l5_pass_b, apply_norm, use_sigmoid):
    # Expected Output shape for image of 224x224 -> 224/2**(3) -> 28x28x3 
    # Input->28x28x256
    # Concat layer
    l4_cat_1 = tf.concat([l4_out_1, l5_pass_a], -1)
    l4_out_2 = MultiConvolutionBlock(256, 3, apply_norm)(l4_cat_1)
    l4_out_3 = MultiConvolutionBlock(256, 3, apply_norm)(l4_out_2) + l4_out_2
    l4_out_4 = MultiConvolutionBlock(256, 3, apply_norm)(l4_out_3) + l4_out_3
    l4_out_5 = MultiConvolutionBlock(256, 3, apply_norm)(l4_out_4)
    # Concat layer
    l4_cat_2 = tf.concat([l4_out_5, l5_pass_b], -1)
    l4_out_6 = MultiConvolutionBlock(256, 3, apply_norm)(l4_cat_2)
    # Upsample for pass
    l4_pass_a = UpSampleConvLayer(128, 3)(l4_out_6)
    l4_pass_b = UpSampleConvLayer(128, 3)(l4_out_6)
    # Final Output
    l4_out_final = ConvLayer(3, kernel_size=3, strides=1, apply_activation=False)(l4_out_6)
    if use_sigmoid == True:
        l4_out_final = keras.activations.sigmoid(l4_out_final)
    else :
        # Extra Values from original TF implementation https://github.com/aiff22/PyNET/blob/16eeb71d94fb0c6ed40a5403ca5efd762974b2bf/model.py#LL40C8-L40C8
        l4_out_final = keras.activations.tanh(l4_out_final) * 0.58 + 0.5

    return l4_out_final,l4_pass_a,l4_pass_b

def level_3(l3_out_1, l4_pass_a, l4_pass_b, apply_norm, use_sigmoid):
    # Expected Output shape for image of 224x224 -> 224/2**(2) -> 56x56x3 
    # Input->56x56x128
    # Concat layer
    l3_cat_1 = tf.concat([l3_out_1, l4_pass_a], -1)
    l3_out_2 = MultiConvolutionBlock(128, 5, apply_norm)(l3_cat_1) + l3_cat_1
    l3_out_3 = MultiConvolutionBlock(128, 5, apply_norm)(l3_out_2) + l3_out_2
    l3_out_4 = MultiConvolutionBlock(128, 5, apply_norm)(l3_out_3) + l3_out_3
    l3_out_5 = MultiConvolutionBlock(128, 5, apply_norm)(l3_out_4)
    # Concat layer
    l3_cat_2 = tf.concat([l3_out_5,l3_out_1,l4_pass_b], -1)
    l3_out_6 = MultiConvolutionBlock(128, 3, apply_norm)(l3_cat_2)
    # Upsample for pass
    l3_pass_a = UpSampleConvLayer(64, 3)(l3_out_6)
    l3_pass_b = UpSampleConvLayer(64, 3)(l3_out_6)
    # Final Output
    l3_out_final = ConvLayer(3, kernel_size=3, strides=1, apply_activation=False)(l3_out_6)
    if use_sigmoid == True:
        l3_out_final = keras.activations.sigmoid(l3_out_final)
    else:
        # Extra Values from original TF implementation https://github.com/aiff22/PyNET/blob/16eeb71d94fb0c6ed40a5403ca5efd762974b2bf/model.py#LL40C8-L40C8
        l3_out_final = keras.activations.tanh(l3_out_final) * 0.58 + 0.5

    return l3_out_final,l3_pass_a,l3_pass_b

def level_2(l2_out_1, l3_pass_a, l3_pass_b, apply_norm, use_sigmoid):
    # Expected Output shape for image of 224x224 -> 224/2**(1) -> 112x112x3 
    # Input->112x112x64
    l2_cat_1 = tf.concat([l2_out_1, l3_pass_a], -1)
    l2_out_2 = MultiConvolutionBlock(64, 5, apply_norm)(l2_cat_1)
    
    # Concat layer
    l2_cat_2 = tf.concat([l2_out_2,l2_out_1], -1)
    l2_out_3 = MultiConvolutionBlock(64, 7, apply_norm)(l2_cat_2) + l2_cat_2
    l2_out_4 = MultiConvolutionBlock(64, 7, apply_norm)(l2_out_3) + l2_out_3
    l2_out_5 = MultiConvolutionBlock(64, 7, apply_norm)(l2_out_4) + l2_out_4
    l2_out_6 = MultiConvolutionBlock(64, 7, apply_norm)(l2_out_5)

    # Concat layer
    l2_cat_3 = tf.concat([l2_out_6,l2_out_1], -1)
    l2_out_7 = MultiConvolutionBlock(64, 5, apply_norm)(l2_cat_3)
    # Concat layer
    l2_cat_4 = tf.concat([l2_out_7,l3_pass_b], -1)
    l2_out_8 = MultiConvolutionBlock(64, 3, apply_norm)(l2_cat_4)
    # Upsample for pass
    l2_pass_a = UpSampleConvLayer(32, 3)(l2_out_8)
    l2_pass_b = UpSampleConvLayer(32, 3)(l2_out_8)
    # Final Output
    l2_out_final = ConvLayer(3, kernel_size=3, strides=1, apply_activation=False)(l2_out_8)
    if use_sigmoid == True:
        l2_out_final = keras.activations.sigmoid(l2_out_final)
    else:
        # Extra Values from original TF implementation https://github.com/aiff22/PyNET/blob/16eeb71d94fb0c6ed40a5403ca5efd762974b2bf/model.py#LL40C8-L40C8
        l2_out_final = keras.activations.tanh(l2_out_final) * 0.58 + 0.5

    return l2_out_final,l2_pass_a,l2_pass_b

def level_1(l1_out_1, l2_pass_a, l2_pass_b, apply_norm, use_sigmoid):
    # Expected Output shape for image of 224x224 -> 224/2**(0) -> 224x224x3 
    # Input->224x224x32
    l1_cat_1 = tf.concat([l1_out_1, l2_pass_a], -1)
    l1_out_2 = MultiConvolutionBlock(32, 5, apply_norm)(l1_cat_1)
    
    # Concat layer
    l1_cat_2 = tf.concat([l1_out_2,l1_out_1], -1)
    l1_out_3 = MultiConvolutionBlock(32, 7, apply_norm)(l1_cat_2)
    l1_out_4 = MultiConvolutionBlock(32, 9, apply_norm)(l1_out_3)
    l1_out_5 = MultiConvolutionBlock(32, 9, apply_norm)(l1_out_4) + l1_out_4
    l1_out_6 = MultiConvolutionBlock(32, 9, apply_norm)(l1_out_5) + l1_out_5
    l1_out_7 = MultiConvolutionBlock(32, 9, apply_norm)(l1_out_6) + l1_out_6
    l1_out_8 = MultiConvolutionBlock(32, 7, apply_norm)(l1_out_7)

    # Concat layer
    l1_cat_3 = tf.concat([l1_out_8,l1_out_1], -1)
    l1_out_9 = MultiConvolutionBlock(32, 5, apply_norm)(l1_cat_3)
    # Concat layer
    l1_cat_4 = tf.concat([l1_out_9,l1_out_1,l2_pass_b], -1)
    l1_out_10 = MultiConvolutionBlock(32, 3, apply_norm)(l1_cat_4)
    # Upsample for Level 0 (448 x 448)
    l1_pass = UpSampleConvLayer(16, 3)(l1_out_10)
    # Final Output
    l1_out_final = ConvLayer(3, kernel_size=3, strides=1, apply_activation=False)(l1_out_10)
    if use_sigmoid == True:
        l1_out_final = keras.activations.sigmoid(l1_out_final)
    else:
        # Extra Values from original TF implementation https://github.com/aiff22/PyNET/blob/16eeb71d94fb0c6ed40a5403ca5efd762974b2bf/model.py#LL40C8-L40C8    
        l1_out_final = keras.activations.tanh(l1_out_final) * 0.58 + 0.5

    return l1_out_final, l1_pass

def level_0(l1_pass, use_sigmoid):
    # Upscaled Image 224x224 -> 448x448
    l0_out_final = ConvLayer(3, kernel_size=3, strides=1, apply_activation=False)(l1_pass)
    if use_sigmoid == True:
        l0_out_final = keras.activations.sigmoid(l0_out_final)
    else:
        # Extra Values from original TF implementation https://github.com/aiff22/PyNET/blob/16eeb71d94fb0c6ed40a5403ca5efd762974b2bf/model.py#LL40C8-L40C8
        l0_out_final = keras.activations.tanh(l0_out_final) * 0.58 + 0.5
    return l0_out_final

def input_shape_check():
    inp = tf.keras.Input((224,224,4))
    l1 = MultiConvolutionBlock(32, 3, True)(inp)
    l2 = keras.layers.MaxPool2D((2,2))(l1)
    l2 = MultiConvolutionBlock(64, 3, True)(l2)
    l3 = keras.layers.MaxPool2D((2,2))(l2)
    l3 = MultiConvolutionBlock(128, 3, True)(l3)
    l4 = keras.layers.MaxPool2D((2,2))(l3)
    l4 = MultiConvolutionBlock(512, 3, True)(l4)
    l5 = keras.layers.MaxPool2D((2,2))(l4)
    return keras.Model(inputs=inp, outputs=[l5], name='shape_check')
