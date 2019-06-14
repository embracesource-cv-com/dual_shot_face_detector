# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: verify.py
@author: danna.li
@time: 2019-06-04 09:57
@description: 
"""
import numpy as np
import keras.layers as KL
from keras import Model
from dual_conf import current_config as conf
from net.dual_shot import test_net
from prepare_data.generator import gen_test, layer_strides, map_size, e_scale, ratio
import os
from prepare_data.model_target import apply_regress, init_anchors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import uuid

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    dets = dets[0:750, :]
    return dets


def eval_model():
    # load model
    weight_path = os.path.join(conf.output_dir, 'weights.h5')
    print('loading trained model from:', weight_path)
    net_in = KL.Input([640, 640, 3], name='image_array')
    ss_cls, ss_regr = test_net(net_in)
    model = Model(inputs=[net_in], outputs=[ss_cls, ss_regr])
    # model.summary()
    model.load_weights(weight_path, by_name=True)

    # load data
    gen = gen_test(conf.batch_size, 'train')
    x_val, y_val = next(gen)

    # make prediction
    ss_cls, ss_regr = model.predict(x_val)
    ss_cls = np.exp(ss_cls) / np.sum(np.exp(ss_cls), axis=0)

    # initialize anchors
    e_anchors = init_anchors(layer_strides, map_size, ratio, e_scale)

    # apply delta
    pred_box = np.empty_like(ss_regr)
    for i in range(len(ss_regr)):
        pred_box[i] = apply_regress(ss_regr[i], e_anchors)

    # top 5k predicted box
    pred_score = ss_cls[..., 1]
    sort_inds = np.argsort(-pred_score)
    pred_score = pred_score[np.arange(pred_score.shape[0])[:, None], sort_inds]
    pred_box = pred_box[np.arange(pred_box.shape[0])[:, None], sort_inds]
    pred_score = pred_score[:, :5000]
    pred_box = pred_box[:, :5000]

    # nms
    final_boxes = []
    for i in range(len(pred_score)):
        box_and_score = np.concatenate([pred_box[i], np.expand_dims(pred_score[i], 1)], axis=1)
        final_box = bbox_vote(box_and_score)
        final_boxes.append(final_box)
    return final_boxes, x_val, y_val


def plot_anchor(img_array, anchor_list):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_array.astype(int))
    for a in anchor_list:
        y1, x1, y2, x2 = [int(i) for i in a]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # plt.show()
    path = os.path.join(conf.output_dir, 'pred_img', str(uuid.uuid1()) + ".jpg")
    plt.savefig(path)


def save_img_result():
    final_boxes, x_val, y_val = eval_model()
    for i in range(len(x_val)):
        plot_anchor(x_val[i], final_boxes[i][:20, :4]) # only visualize top 20 boxes


if __name__ == '__main__':
    save_img_result()
