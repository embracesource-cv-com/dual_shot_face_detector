# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/25 
@file: dual_conf.py
@description:
"""


class Config(object):
    data_path = '/home/dataset/face_recognize/face_detect'
    output_dir = '/opt/code/face/dual_shot/train_out'
    num_class = 2
    net_in_size = 640
    num_anchor = 34125
    num_image = 12876
    batch_size = 4
    epochs = 20
    steps_per_epoch = 3000  # 约3000/epoch
    gpu_index = "1"
    num_train_anchor = 128
    iou_thread = 0.4

    lr = 0.001
    weight_decay = 0.0005
    base_net='resnet50'  # can be 'resnet50','resnet101'

    continue_training = True
    weights_to_transfer = '/home/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# 当前生效配置
current_config = Config()
