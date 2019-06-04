# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/29 
@file: generator.py
@description:
"""

import numpy as np
from dual_conf import current_config as conf
from prepare_data.widerface import WIDERFaceDetection, WIDERFaceAnnotationTransform
from prepare_data.augment import Augmentation
from prepare_data.model_target import init_anchors, cal_target

layer_strides = np.array([4, 8, 16, 32, 64, 128])
map_size = np.array([160, 80, 40, 20, 10, 5])
e_scale = np.array([16, 32, 64, 128, 256, 512])
o_scale = (e_scale / 2).astype('int')
ratio = 1 / 1.5


def image_reader(test_set):
    index = np.random.randint(0, conf.num_image, 1)[0]
    # print('batch_index:', index)
    image = test_set.pull_image(index)
    _, gts = test_set.pull_anno(index)
    gts = np.array(gts)
    labels = np.full(gts.shape[0], 0)
    if len(gts) == 0:
        image_reader(test_set)
    return image, gts, labels


def augment(image, gts, labels, test_set):
    to_aug = Augmentation(conf.net_in_size)
    sub_img, gt_in_crop, sparse_label = to_aug(image, gts, labels)
    if len(gt_in_crop) == 0:
        image, gts, labels = image_reader(test_set)
        augment(image, gts, labels, test_set)
    return sub_img, gt_in_crop, sparse_label


def gen_data(batch_size, phase='train'):
    test_set = WIDERFaceDetection(conf.data_path, phase, None, WIDERFaceAnnotationTransform())
    e_anchors = init_anchors(layer_strides, map_size, ratio, e_scale)
    o_anchors = init_anchors(layer_strides, map_size, ratio, o_scale)
    while True:
        batch_img = []
        batch_gt = []
        e_reg_targets, e_ind_trains = [], []
        o_reg_targets, o_ind_trains = [], []
        for i in range(batch_size):
            image, gts, labels = image_reader(test_set)
            sub_img, gt_in_crop, sparse_label = augment(image, gts, labels, test_set)
            # print('num_gt:', len(gt_in_crop))
            e_reg_target, e_ind_train = cal_target(gt_in_crop, e_anchors, iou_thread=conf.iou_thread,
                                                   train_anchors=conf.num_train_anchor)
            o_reg_target, o_ind_train = cal_target(gt_in_crop, o_anchors, iou_thread=conf.iou_thread,
                                                   train_anchors=conf.num_train_anchor)
            e_reg_targets.append(e_reg_target)
            e_ind_trains.append(e_ind_train)
            o_reg_targets.append(o_reg_target)
            o_ind_trains.append(o_ind_train)
            batch_img.append(sub_img)
            batch_gt.append(gt_in_crop)
        e_reg_targets, e_ind_trains = [np.array(i) for i in [e_reg_targets, e_ind_trains]]
        o_reg_targets, o_ind_trains = [np.array(i) for i in [o_reg_targets, o_ind_trains]]
        batch_img = np.array(batch_img)
        yield [batch_img, e_reg_targets, e_ind_trains, o_reg_targets, o_ind_trains], []


def gen_test(batch_size, phase='train'):
    test_set = WIDERFaceDetection(conf.data_path, phase, None, WIDERFaceAnnotationTransform())
    while True:
        batch_img = []
        batch_gt = []
        for i in range(batch_size):
            image, gts, labels = image_reader(test_set)
            sub_img, gt_in_crop, sparse_label = augment(image, gts, labels, test_set)
            batch_img.append(sub_img)
            batch_gt.append(gt_in_crop)
        batch_img = np.array(batch_img)
        batch_gt = np.array(batch_gt)
        yield batch_img, batch_gt
