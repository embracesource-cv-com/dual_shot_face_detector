# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/28 
@file: model_target.py
@description:
"""

import numpy as np
from sklearn.utils import shuffle
from dual_conf import current_config as conf


def generate_anchors(base_size, ratios, scales):
    """
    根据基准尺寸、长宽比、缩放比生成边框
    :param base_size: anchor的base_size,如：128
    :param ratios: 长宽比 shape:(M,),如：[0.5,1,2]
    :param scales: 缩放比 shape:(N,),如：[1,2,4]
    :return: （N*M,(x1,y1,x2,y2))
    """
    ratios = np.expand_dims(np.array(ratios), axis=1)  # (N,1)
    scales = np.expand_dims(np.array(scales), axis=0)  # (1,M)
    # 计算高度和宽度，形状为(N,M)
    h = np.sqrt(ratios) * scales * base_size
    w = 1.0 / np.sqrt(ratios) * scales * base_size
    # reshape为（N*M,1)
    h = np.reshape(h, (-1, 1))
    w = np.reshape(w, (-1, 1))

    return np.hstack([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h])


def shift(base_anchors, shift_shape, strides):
    """
    生成所有anchors
    :param base_anchors: 基准anchors，[N,(y1,x1,y2,x2)]
    :param shift_shape: tuple或list (h,w), 最终的anchors个数为h*w*N
    :param strides:tuple或list (stride_h,stride_w),
    :return:
    """
    # 计算中心点在原图坐标
    center_y = strides[0] * (0.5 + np.arange(shift_shape[0]))
    center_x = strides[1] * (0.5 + np.arange(shift_shape[1]))

    center_x, center_y = np.meshgrid(center_x, center_y)  # (h,w)

    center_x = np.reshape(center_x, (-1, 1))  # (h*w,1)
    center_y = np.reshape(center_y, (-1, 1))  # (h*w,1)

    dual_center = np.concatenate([center_y, center_x, center_y, center_x], axis=1)  # (h*w,4)

    # 中心点和基准anchor合并
    base_anchors = np.expand_dims(base_anchors, axis=0)  # (1,N,4)
    dual_center = np.expand_dims(dual_center, axis=1)  # (h*w,1,4)
    anchors = base_anchors + dual_center  # (h*w,N,4)
    # 打平返回
    return np.reshape(anchors, (-1, 4))  # (h*w,4)


def init_anchors(layer_strides, map_size, ratio, scale):
    all_anchors = []
    for i in range(len(layer_strides)):
        base_anchor = generate_anchors(base_size=1, ratios=ratio, scales=scale[i])
        layer_anchors = shift(base_anchor, [map_size[i], map_size[i]], [layer_strides[i], layer_strides[i]])
        all_anchors.append(layer_anchors)
    all_anchors = np.vstack(all_anchors)
    all_anchors[all_anchors < 0] = 0
    all_anchors[all_anchors > conf.net_in_size] = conf.net_in_size
    return all_anchors


def iou_np(boxes_a, boxes_b):
    """
    numpy 计算IoU
    :param boxes_a: (N,4)
    :param boxes_b: (M,4)
    :return:  IoU (N,M)
    """
    # 扩维
    boxes_a = np.expand_dims(boxes_a, axis=1)  # (N,1,4)
    boxes_b = np.expand_dims(boxes_b, axis=0)  # (1,M,4)

    # 分别计算高度和宽度的交集
    overlap_h = np.maximum(0.0,
                           np.minimum(boxes_a[..., 2], boxes_b[..., 2]) -
                           np.maximum(boxes_a[..., 0], boxes_b[..., 0]))  # (N,M)

    overlap_w = np.maximum(0.0,
                           np.minimum(boxes_a[..., 3], boxes_b[..., 3]) -
                           np.maximum(boxes_a[..., 1], boxes_b[..., 1]))  # (N,M)

    # 交集
    overlap = overlap_w * overlap_h

    # 计算面积
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou


def regress_target(anchors, gt_boxes):
    """
    计算回归目标
    :param anchors: [N,(y1,x1,y2,x2)]
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :return: [N,(dy, dx, dh, dw)]
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    # 中心点
    center_y = (anchors[:, 2] + anchors[:, 0]) * 0.5
    center_x = (anchors[:, 3] + anchors[:, 1]) * 0.5
    gt_center_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) * 0.5
    gt_center_x = (gt_boxes[:, 3] + gt_boxes[:, 1]) * 0.5

    # 回归目标
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dh = np.log(gt_h / h)
    dw = np.log(gt_w / w)

    target = np.stack([dy, dx, dh, dw], axis=1)
    target /= np.array([0.1, 0.1, 0.2, 0.2])
    # target = tf.where(tf.greater(target, 100.0), 100.0, target)
    return target


def apply_regress(deltas, anchors):
    """
    应用回归目标到边框
    :param deltas: 回归目标[N,(dy, dx, dh, dw)]
    :param anchors: anchor boxes[N,(y1,x1,y2,x2)]
    :return:
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cy = (anchors[:, 2] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 3] + anchors[:, 1]) * 0.5

    # 回归系数
    deltas *= np.array(([0.1, 0.1, 0.2, 0.2]))
    dy, dx, dh, dw = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 中心坐标回归
    cy += dy * h
    cx += dx * w
    # 高度和宽度回归
    h *= np.exp(dh)
    w *= np.exp(dw)

    # 转为y1,x1,y2,x2
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5

    return np.stack([y1, x1, y2, x2], axis=1)


def cal_target(gts=None, anchors=None, iou_thread=0.4, train_anchors=10):
    # calculate iou matrix
    iou = iou_np(gts, anchors)
    # print('max_iou:', np.max(iou))
    # get positive anchors which have iou bigger than 0.7 with any GT
    iou_sign = np.where(iou > iou_thread, np.ones_like(iou), np.zeros_like(iou))
    anchor_cls = np.argmax(iou_sign, axis=0)
    # get positive anchors which has the highest iou with any GT,while the highest iou<0.7
    row_max_ind = np.argmax(iou, axis=1)  # shape(2,)
    # combine 2 types of positive anchors
    anchor_cls = [anchor_cls[i] + 1 if i in row_max_ind else anchor_cls[i] for i in range(len(anchor_cls))]
    anchor_cls = np.array(anchor_cls)
    anchor_cls[anchor_cls > 1] = 1
    # get non-positive index where iou with gt is <0.3
    bad_ind = np.where(np.equal(anchor_cls, 0))[0]
    bad_iou = iou[:, bad_ind]
    bad_iou = np.max(bad_iou, axis=0)
    bad_neg_ind = np.where(np.greater(0.3, bad_iou))[0]
    neg_ind = bad_ind[bad_neg_ind]

    # randomly select positive anchors
    pos_ind = np.where(np.not_equal(anchor_cls, 0))[0]
    seed = np.random.randint(0, 1000, 1)[0]
    pos_ind_chose = shuffle(pos_ind, random_state=seed)[0:train_anchors // 2]
    neg_ind_chose = shuffle(neg_ind, random_state=seed)[0:train_anchors // 2]

    # calculate regression target for positive anchors
    pos_a_chose = anchors[pos_ind_chose]
    col_max_ind = np.argmax(iou, axis=0)
    gt_for_anchor = np.where(anchor_cls > 0, col_max_ind, np.full_like(anchor_cls, -1))
    gt_ind_to_pos_a = gt_for_anchor[pos_ind_chose]
    gt_to_pos_a = gts[gt_ind_to_pos_a]
    pos_reg_target = regress_target(anchors=pos_a_chose, gt_boxes=gt_to_pos_a)
    # padding regression target
    pos_reg_target = np.pad(pos_reg_target, [[0, 0], [0, 1]], mode='constant', constant_values=1)
    pos_train_anchor = train_anchors - pos_reg_target.shape[0]
    reg_target = np.pad(pos_reg_target, [[0, pos_train_anchor], [0, 0]], mode='constant', constant_values=0)

    # get index of training anchors
    pos_ind_train = np.pad(np.expand_dims(pos_ind_chose, 1), [[0, 0], [0, 1]], mode='constant', constant_values=1)
    neg_ind_train = np.pad(np.expand_dims(neg_ind_chose, 1), [[0, 0], [0, 1]], mode='constant', constant_values=-1)
    a_ind_train = np.concatenate([pos_ind_train, neg_ind_train], axis=0)
    a_ind_train = np.pad(a_ind_train, [[0, train_anchors - a_ind_train.shape[0]], [0, 0]], mode='constant')
    # cls_target = a_ind_train[:, -1]
    # print('train_num distribution:', np.unique(a_ind_train[:, -1], return_counts=True))
    return reg_target, a_ind_train

# if __name__ == '__main__':
#     layer_strides = np.array([4, 8, 16, 32, 64, 128])
#     map_size = np.array([160, 80, 40, 20, 10, 5])
#     e_scale = np.array([16, 32, 64, 128, 256, 512])
#     o_scale = (e_scale / 2).astype('int')
#     ratio = 1.5
#     o_anchors = init_anchors(layer_strides, map_size, ratio, scale=o_scale)
#     print(o_anchors.shape)
#     e_anchors = init_anchors(layer_strides, map_size, ratio, scale=e_scale)
#     print(o_anchors.shape)
