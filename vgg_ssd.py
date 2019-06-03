# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/24 
@file: vgg_ssd.py
@description:
"""
import keras.layers as KL
from keras import Model
from dual_conf import current_config as conf
import keras
from loss_func import progressive_anchor_loss


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


def classifier(x, num_anchor, name):
    cls = KL.Conv2D(conf.num_class, (1, 1), activation='softmax', padding='same', name='cls_' + name)(x)
    regr = KL.Conv2D(4, (1, 1), activation='linear', padding='same', name='regr_' + name)(x)
    cls = KL.Reshape((num_anchor, conf.num_class), name='reshape_cls_' + name)(cls)
    regr = KL.Reshape((num_anchor, 4), name='reshape_regr_' + name)(regr)
    return cls, regr


def detector(conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2, name):
    cls1, regr1 = classifier(conv3_3, 25600, 'conv3_3' + name)
    cls2, regr2 = classifier(conv4_3, 6400, 'conv4_3' + name)
    cls3, regr3 = classifier(conv5_3, 1600, 'conv5_3' + name)
    cls4, regr4 = classifier(conv_fc7, 400, 'fc_7' + name)
    cls5, regr5 = classifier(conv6_2, 100, 'conv6_2' + name)
    cls6, regr6 = classifier(conv7_2, 25, 'conv7_2' + name)
    cls = KL.Concatenate(axis=1, name='cls_concat_' + name)([cls1, cls2, cls3, cls4, cls5, cls6])
    regr = KL.Concatenate(axis=1, name='regr_concat_' + name)([regr1, regr2, regr3, regr4, regr5, regr6])
    return cls, regr


def group_channel_conv(x_in, name):
    x1_1 = KL.Conv2D(256, (3, 3), dilation_rate=(1, 1), activation='relu', padding='same', name='dilated_1_1' + name)(
        x_in)
    x2_1 = KL.Conv2D(256, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='dilated_2_1' + name)(
        x_in)
    x2_2 = KL.Conv2D(128, (3, 3), dilation_rate=(1, 1), activation='relu', padding='same', name='dilated_2_2' + name)(
        x2_1)
    x3_1 = KL.Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='dilated_3_1' + name)(
        x2_1)
    x3_2 = KL.Conv2D(128, (3, 3), dilation_rate=(1, 1), activation='relu', padding='same', name='dilated_3_2' + name)(
        x3_1)
    return KL.Concatenate()([x1_1, x2_2, x3_2])


def feature_enhance(x_current, x_deeper, channel, name):
    x_current = KL.Conv2D(channel, (1, 1), activation='relu', padding='same', name='current_norm_' + name)(x_current)
    x_deeper = KL.Conv2D(channel, (1, 1), activation='relu', padding='same', name='deeper_norm_' + name)(x_deeper)
    x_deeper = KL.UpSampling2D(size=(2, 2))(x_deeper)
    x_combine = KL.Multiply(name='element_dot_' + name)([x_current, x_deeper])
    x = group_channel_conv(x_combine, name)
    return x


def feature_enhance_module(conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2):
    conv3_3_ef = feature_enhance(conv3_3, conv4_3, 256, 'conv3_3')
    conv4_3_ef = feature_enhance(conv4_3, conv5_3, 256, 'conv4_3')
    conv5_3_ef = feature_enhance(conv5_3, conv_fc7, 256, 'conv5_3')
    conv_fc7_ef = feature_enhance(conv_fc7, conv6_2, 256, 'conv_fc7')
    conv6_2_ef = feature_enhance(conv6_2, conv7_2, 256, 'conv6_2')
    conv7_2_ef = conv7_2
    return conv3_3_ef, conv4_3_ef, conv5_3_ef, conv_fc7_ef, conv6_2_ef, conv7_2_ef


def whole_net(x_in, y_e_reg, y_e_ind, y_o_reg, y_o_ind):
    conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2 = extend_vgg(x_in)
    # first shot
    fs_cls, fs_regr = detector(conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2, 'fs')
    # second shot
    conv3_3_ef, conv4_3_ef, conv5_3_ef, conv_fc7_ef, conv6_2_ef, conv7_2_ef \
        = feature_enhance_module(conv3_3, conv4_3, conv5_3, conv_fc7, conv6_2, conv7_2)
    ss_cls, ss_regr = detector(conv3_3_ef, conv4_3_ef, conv5_3_ef, conv_fc7_ef, conv6_2_ef, conv7_2_ef, 'ss')
    pal = keras.layers.Lambda(lambda x: progressive_anchor_loss(*x), name='PAL')([y_e_reg, y_e_ind, y_o_reg, y_o_ind,
                                                                                  fs_cls, fs_regr, ss_cls, ss_regr])
    return pal


def net_test():
    net_in = KL.Input([640, 640, 3], name='image_array')
    y_e_reg = KL.Input((34125, 5), name='e_reg')
    y_e_ind = KL.Input((34125, 2), name='e_train_ind')
    y_o_reg = KL.Input((34125, 5), name='o_reg')
    y_o_ind = KL.Input((34125, 2), name='o_train_reg')
    pal = whole_net(net_in, y_e_reg, y_e_ind, y_o_reg, y_o_ind)
    model = Model(inputs=[net_in, y_e_reg, y_e_ind, y_o_reg, y_o_ind], outputs=pal)
    model.summary()
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)


if __name__ == '__main__':
    net_test()