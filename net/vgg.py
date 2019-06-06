# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: vgg.py
@author: danna.li
@time: 2019-06-06 11:17
@description: 
"""
import keras.layers as KL


def extend_vgg(x_in):
    # VGG
    # Block 1
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x_in)
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    conv3_3 = x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    conv4_3 = x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    conv5_3 = x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Extended Layers
    x = KL.Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv_fc6')(x)
    conv_fc7 = x = KL.Conv2D(1024, (1, 1), activation='relu', padding='same', name='conv_fc7')(x)
    x = KL.Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(x)
    conv6_2 = x = KL.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(x)
    x = KL.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(x)
    conv7_2 = KL.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_2')(x)
    return conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2