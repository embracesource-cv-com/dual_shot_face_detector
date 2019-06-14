# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/29 
@file: generator.py
@description:
"""

import numpy as np
from dual_conf import current_config as conf
from prepare_data.widerface import WIDERFaceDetection
from prepare_data.augment import Augmentation
from prepare_data.model_target import init_anchors, cal_target

layer_strides = np.array([4, 8, 16, 32, 64, 128])
map_size = np.array([160, 80, 40, 20, 10, 5])
e_scale = np.array([16, 32, 64, 128, 256, 512])
o_scale = (e_scale / 2).astype('int')
ratio = 1.5


def image_reader(test_set):
    index = np.random.randint(0, len(test_set), 1)[0]
    # print('batch_index:', index)
    image, gts, labels = test_set.pull_item(index)
    if len(gts) == 0:
        image_reader(test_set)
    return image, gts, labels


def augment(image, gts, labels, test_set):
    to_aug = Augmentation(conf.net_in_size)
    sub_img, gt_in_crop, sparse_label = to_aug(image, gts, labels)
    if len(gt_in_crop) == 0:
        image, gts, labels = image_reader(test_set)
        sub_img, gt_in_crop, sparse_label = augment(image, gts, labels, test_set)
    return sub_img, gt_in_crop, sparse_label


def gen_data(batch_size, phase='train'):
    test_set = WIDERFaceDetection(conf.data_path, phase, None, None)
    e_anchors = init_anchors(layer_strides, map_size, ratio, e_scale)
    o_anchors = init_anchors(layer_strides, map_size, ratio, o_scale)
    while True:
        batch_img = []
        batch_gt = []
        e_reg_targets, e_cls_targets = [], []
        o_reg_targets, o_cls_targets = [], []
        for i in range(batch_size):
            image, gts, labels = image_reader(test_set)
            sub_img, gt_in_crop, sparse_label = augment(image, gts, labels, test_set)
            # print('num_gt:', len(gt_in_crop))
            gt_in_crop = gt_in_crop[:, np.array([1, 0, 3, 2])]  # change x1,y1,x2,y2 to y1,x1,y2,x2
            e_reg_target, e_cls_target = cal_target(gt_in_crop, e_anchors, conf.iou_thread, 100)
            o_reg_target, o_cls_target = cal_target(gt_in_crop, o_anchors, conf.iou_thread, 100)
            e_reg_targets.append(e_reg_target)
            e_cls_targets.append(e_cls_target)
            o_reg_targets.append(o_reg_target)
            o_cls_targets.append(o_cls_target)
            batch_img.append(sub_img)
            batch_gt.append(gt_in_crop)

        e_reg_targets, e_cls_targets = [np.array(i) for i in [e_reg_targets, e_cls_targets]]
        o_reg_targets, o_cls_targets = [np.array(i) for i in [o_reg_targets, o_cls_targets]]
        batch_img = np.array(batch_img)
        yield [batch_img, e_reg_targets, e_cls_targets, o_reg_targets, o_cls_targets], []


def gen_test(batch_size, phase='val'):
    test_set = WIDERFaceDetection(conf.data_path, phase, None, None)
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


def load_dataset(phase='val'):
    test_set = WIDERFaceDetection(conf.data_path, phase, None, None)
    batch_img = []
    batch_gt = []
    for i in range(len(test_set)):
        image, gts, labels = image_reader(test_set)
        sub_img, gt_in_crop, sparse_label = augment(image, gts, labels, test_set)
        batch_img.append(sub_img)
        batch_gt.append(gt_in_crop)
    batch_img = np.array(batch_img)
    batch_gt = np.array(batch_gt)
    return batch_img, batch_gt
