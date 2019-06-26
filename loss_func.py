# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/25
@file: loss_func.py
@description:
"""

import tensorflow as tf
import keras.backend as K
from dual_conf import current_config as conf


def log_sum_exp(cls_target, predict_logits):
    """un-averaged confidence loss across all examples in a batch.
    :param cls_target: 2-d tensor,[num_anchors,2 ]; 1,0 for pos,neg anchors respectively
    :param predict_logits: 2-d tensor,[num_anchors,2] 2 for fg and bg
    :return: loss,1-D tensor
    """
    logit_max = tf.reduce_max(predict_logits)
    loss = tf.log(tf.reduce_sum(tf.exp(predict_logits - logit_max), 1, keepdims=True)) + logit_max
    cls_indice = tf.where(cls_target >= 1)
    anchor_cls_logit = tf.gather_nd(predict_logits, cls_indice)
    loss = tf.squeeze(loss)
    loss = loss - anchor_cls_logit
    return loss


def hard_neg_mining(cls_target, predict_logits):
    """
    select top 80 negtive anchors the contribute most loss to do the backward pass
    :param cls_target:2-d tensor,[num_anchors,2 ]; 1,0 for pos,neg anchors respectively
    :param predict_logits: 2-d tensor,[num_anchors,2] 2 for fg and bg
    :return:
    """
    loss_to_rank = log_sum_exp(cls_target, predict_logits)
    loss_to_rank = tf.cast(cls_target[:, 0], tf.float64) * tf.cast(loss_to_rank,
                                                                   tf.float64)  # set loss for pos anchor to 0
    neg_num = tf.cast(tf.reduce_sum(cls_target[:, -1]) * 3, tf.int32)
    _, neg_ind = tf.nn.top_k(loss_to_rank, neg_num)
    pos_ind = tf.squeeze(tf.where(cls_target[:, 1] >= 1))
    pos_ind = tf.cast(pos_ind, tf.int64)
    neg_ind = tf.cast(neg_ind, tf.int64)
    all_ind = tf.concat([pos_ind, neg_ind], axis=0)
    cls_target = tf.gather(cls_target, all_ind)
    predict_logits = tf.gather(predict_logits, all_ind)
    return cls_target, predict_logits


def cls_loss(cls_target, predict_logits):
    """
    :param cls_target:2-d array,[batch_num,num_anchors]; 1,-1,0 for pos,neg and un-train anchors respectively
    :param predict_logits: 3-d array,[batch_num,num_anchors,2] fg or bg
    :return: classification loss of training anchors
    """
    # remove un-trained anchors from cls_target and make cls_target to one-hot format
    train_indices = tf.where(tf.not_equal(cls_target, 0))
    cls_target = tf.gather_nd(cls_target, train_indices)
    # change negative tag from -1 to 0
    cls_target = tf.where(cls_target > 0, cls_target, tf.zeros_like(cls_target))
    cls_target = tf.cast(cls_target, tf.int64)
    cls_target = tf.one_hot(cls_target, depth=conf.num_class)

    # remove un-trained anchors from pred logit
    logit0 = tf.gather_nd(predict_logits[..., 0], train_indices)
    logit1 = tf.gather_nd(predict_logits[..., 1], train_indices)
    predict_logits = tf.stack([logit0, logit1], axis=1)
    # predict_logits = tf.cast(predict_logits, dtype=tf.float32)

    # hard negative anchor mining
    if conf.hard_negative_mining:
        cls_target, predict_logits = hard_neg_mining(cls_target, predict_logits)

    #  calculate the loss
    print(cls_target, predict_logits)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=cls_target, logits=predict_logits)
    loss = K.mean(loss)
    return loss


def regr_loss(reg_target, predict_deltas, cls_target):
    """
    :param reg_target: 2-d array,[batch_num,num_anchors,(dy,dx,dz,dh)]fg or bg
    :param predict_deltas: 2-d array,[batch_num,num_anchors,(dy,dx,dz,dh)] fg or bg
    :param cls_target: 2-d array,[batch_num,num_anchors]; 1,-1,0 for pos,neg and un-train anchors respectively
    :return:
    """
    # remove un-trained anchors
    pos_indice = tf.where(tf.equal(cls_target, 1))
    reg_target = tf.gather_nd(reg_target, pos_indice)
    pred_deltas = tf.gather_nd(predict_deltas, pos_indice)
    diff = tf.abs(reg_target - pred_deltas)
    smooth_loss = tf.where(diff < 1, tf.pow(diff, 2) * 0.5, diff - 0.5)
    loss = K.mean(smooth_loss)
    return loss


def progressive_anchor_loss(e_reg_targets, e_cls_targets, o_reg_targets, o_cls_targets,
                            fs_cls, fs_regr, ss_cls, ss_regr):
    fs_cls_loss = cls_loss(o_cls_targets, fs_cls)
    ss_cls_loss = cls_loss(e_cls_targets, ss_cls)
    fs_regr_loss = regr_loss(o_reg_targets, fs_regr, o_cls_targets)
    ss_regr_loss = regr_loss(e_reg_targets, ss_regr, e_cls_targets)

    # fs_cls_loss = K.print_tensor(fs_cls_loss, message='fs_cls_loss = ')
    # ss_cls_loss = K.print_tensor(ss_cls_loss, message='ss_cls_loss = ')
    # fs_regr_loss = K.print_tensor(fs_regr_loss, message='fs_regr_loss = ')
    # ss_regr_loss = K.print_tensor(ss_regr_loss, message='ss_regr_loss = ')

    total_loss = fs_cls_loss + ss_cls_loss + fs_regr_loss + ss_regr_loss
    # total_loss = K.print_tensor(total_loss, message='total_loss = ')
    return total_loss


if __name__ == '__main__':
    import numpy as np

    cls_target = np.random.randint(-1, 2, [2, 8])
    predict_logits = np.random.randint(-1, 10, [2, 8, 2])
    cls_target = tf.constant(cls_target)
    predict_logits = tf.constant(predict_logits, tf.float64)
    loss = cls_loss(cls_target, predict_logits)
    to_run = [loss]
    sess = tf.Session()
    for i in to_run:
        print(i, '\n', sess.run(i))
