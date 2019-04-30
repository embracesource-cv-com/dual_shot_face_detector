# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/25
@file: loss_func.py
@description:
"""

import tensorflow as tf
import keras.backend as K


def cls_loss(true_cls_ids, predict_cls_ids):
    """
    :param true_cls_ids: 3-d array,(batch_num,rpn_train_anchors,(train_anchor_idx,cls_tag))
    :param predict_cls_ids: 3-d array,(batch_num,anchors_num,2) fg or bg
    :return: classification loss of selected anchors
    """
    # remove padding from true class
    train_indices = tf.where(tf.not_equal(true_cls_ids[..., -1], 0))  # 参与训练的indice
    true_cls = tf.gather_nd(true_cls_ids[..., 1], train_indices)
    true_cls = tf.where(true_cls > 0, tf.ones_like(true_cls), tf.zeros_like(true_cls))  # 在one_hot前，将负样本label-1改为0
    true_cls = tf.cast(true_cls, dtype=tf.int32)
    true_cls = tf.one_hot(true_cls, depth=2)

    # remove padding anchors and un-trained anchors from pred logit
    logit0 = tf.gather_nd(predict_cls_ids[..., 0], train_indices)
    logit1 = tf.gather_nd(predict_cls_ids[..., 1], train_indices)
    logit = tf.stack([logit0, logit1], axis=1)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_cls, logits=logit)
    loss = K.mean(loss)
    return loss


def regr_loss(deltas, predict_deltas, indices):
    """
    :param deltas:
    :param predict_deltas:
    :param indices: 3-d array,(batch_num,rpn_train_anchors,(train_anchor_idx,cls_tag))
    :return:
    """
    pos_indice = tf.where(tf.equal(indices[..., -1], 1))
    true_deltas = tf.gather_nd(deltas, pos_indice)
    true_deltas = true_deltas[..., 0:4]

    pos_anchor_ind = tf.gather_nd(indices[..., 0], pos_indice)
    pos_anchor_ind = tf.cast(pos_anchor_ind, tf.int64)
    batch_index = pos_indice[..., 0]
    pred_indice = tf.stack([batch_index, pos_anchor_ind], axis=1)
    pred_deltas = tf.gather_nd(predict_deltas, pred_indice)
    diff = tf.abs(true_deltas - pred_deltas)
    smooth_loss = tf.where(diff < 1, tf.pow(diff, 2) * 0.5, diff - 0.5)
    loss = K.mean(smooth_loss)
    return loss


def progressive_anchor_loss(e_reg_targets, e_ind_trains, o_reg_targets, o_ind_trains,
                            fs_cls, fs_regr, ss_cls, ss_regr):
    fs_cls_loss = cls_loss(o_ind_trains, fs_cls)
    ss_cls_loss = cls_loss(e_ind_trains, ss_cls)
    fs_regr_loss = regr_loss(o_reg_targets, fs_regr, o_ind_trains)
    ss_regr_loss = regr_loss(e_reg_targets, ss_regr, e_ind_trains)

    total_loss = fs_cls_loss + ss_cls_loss + fs_regr_loss + ss_regr_loss
    return total_loss
