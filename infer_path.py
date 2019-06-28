# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: infer_path.py
@author: danna.li
@time: 2019-06-28 17:45
@description: 
"""
from os.path import join
from glob import glob
import numpy as np
import os
from tqdm import tqdm
from prepare_data.model_target import apply_regress, init_anchors
from dual_conf import current_config as conf
from prepare_data.generator import layer_strides, map_size, e_scale, ratio
from predict_dual import load_model, softmax, bbox_vote, adjust_gt, plot_anchor
import cv2


def load_image_paths(path):
    image_list = []
    for ext in ('*.png', '*.jpg'):
        image_list.extend(glob(join(path, ext)))
    print(image_list)
    return image_list


def eval_path(path):
    image_list = load_image_paths(path)
    model = load_model()
    e_anchors = init_anchors(layer_strides, map_size, ratio, e_scale)

    for index in tqdm(range(len(image_list))):
        img_path = image_list[index]
        image = cv2.imread(img_path)
        plot_img = image.copy()
        origin_width, origin_height = image.shape[1], image.shape[0]

        image = cv2.resize(image, dsize=(conf.net_in_size, conf.net_in_size), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, 0)

        # make prediction
        ss_cls, ss_regr = model.predict(image)
        ss_cls, ss_regr = np.squeeze(ss_cls), np.squeeze(ss_regr)
        ss_cls = softmax(ss_cls)

        # apply delta
        pred_box = apply_regress(ss_regr, e_anchors)

        # select top 5k predicted box
        pred_score = ss_cls[..., 1]
        sort_inds = np.argsort(-pred_score)
        pred_score = pred_score[sort_inds]
        pred_box = pred_box[sort_inds]
        pred_score = pred_score[:5000]
        pred_box = pred_box[:5000, :]

        # nms
        box_and_score = np.concatenate([pred_box, np.expand_dims(pred_score, 1)], axis=1)
        pred_box = bbox_vote(box_and_score)

        pred_box = pred_box[:, :4]
        pred_box = adjust_gt(pred_box, origin_width, origin_height, conf.net_in_size, conf.net_in_size)

        # save result
        img_name = os.path.split(img_path)[-1]
        save_path = os.path.join(path, 'result', img_name)
        plot_anchor(plot_img, pred_box, save_path)

    print('[INFO] Test finished!')


if __name__ == '__main__':
    path_to_infer = './img'
    eval_path(path_to_infer)
