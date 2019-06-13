# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: predict_dual.py
@author: danna.li
@time: 2019-06-13 10:59
@description: 
"""

import numpy as np
import keras.layers as KL
from keras import Model
from dual_conf import current_config as conf
from net.dual_shot import test_net
from prepare_data.generator import layer_strides, map_size, e_scale, ratio
import os
from prepare_data.model_target import apply_regress, init_anchors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import uuid
from prepare_data.widerface import WIDERFaceDetection
from prepare_data.augment import Resize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def softmax(logit):
    assert len(logit.shape) == 2
    row_max = np.max(logit, axis=1)
    row_max = row_max[:, np.newaxis]
    logit = logit - row_max  # make logit negtive to avoid infinity of e_x
    e_x = np.exp(logit)
    row_sum = np.sum(e_x, axis=1)
    row_sum = row_sum[:, np.newaxis]
    return e_x / row_sum


def load_model():
    weight_path = os.path.join(conf.output_dir, 'weights.h5')
    print('loading trained model from:', weight_path)
    net_in = KL.Input([640, 640, 3], name='image_array')
    ss_cls, ss_regr = test_net(net_in)
    model = Model(inputs=[net_in], outputs=[ss_cls, ss_regr])
    # model.summary()
    model.load_weights(weight_path, by_name=True)
    return model


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = None
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[dets[:, -1] > 0.5]  # ignore boxes with scores lower than 0.5
    dets = dets[0:750, :4]
    return dets


def plot_anchor(img_array, anchor_list, path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_array.astype(int))
    for a in anchor_list:
        y1, x1, y2, x2 = [int(i) for i in a]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # plt.show()
    plt.savefig(path)
    plt.close()


def eval_widerface():
    test_set = WIDERFaceDetection(conf.data_path, 'val', None, None)
    model = load_model()
    e_anchors = init_anchors(layer_strides, map_size, ratio, e_scale)

    for index in range(len(test_set)):
        image, gts, labels = test_set.pull_item(index)
        to_resize = Resize(conf.net_in_size)
        image, gts, labels = to_resize(image, gts, labels)
        plot_img = image.copy()
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
        pred_label = np.full_like(pred_score, 1)

        img_path = test_set.pull_image_name(index)
        img_name = os.path.split(img_path)[-1]
        save_path = os.path.join(conf.output_dir, 'widerface_val_result', '{}.npz'.format(img_name))
        np.savez(save_path, gt_boxes=gts, gt_labels=labels,
                 pred_boxes=pred_box, pred_labels=pred_label,pred_scores=pred_score)
        save_path = os.path.join(conf.output_dir, 'widerface_val_result',img_name)
        plot_anchor(plot_img, pred_box, save_path)
        print('[INFO] {}.npz saved...'.format(img_name))

    print('[INFO] Test finished!')


if __name__ == '__main__':
    eval_widerface()
