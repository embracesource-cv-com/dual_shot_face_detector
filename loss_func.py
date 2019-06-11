# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/25
@file: loss_func.py
@description:
"""

import tensorflow as tf
import keras.backend as K


def cls_loss(cls_target, predict_logits):
    """
    :param cls_target:2-d array,[batch_num,num_anchors]; 1,-1,0 for pos,neg and un-train anchors respectively
    :param predict_logits: 2-d array,[batch_num,num_anchors] fg or bg
    :return: classification loss of training anchors
    """
    # remove un-trained anchors from cls_target
    print(cls_target)
    train_indices = tf.where(tf.not_equal(cls_target, 0))
    cls_target = tf.gather_nd(cls_target, train_indices)

    # change negative tag from -1 to 0, and make cls_target to one-hot format
    cls_target = tf.where(cls_target > 0, cls_target, tf.zeros_like(cls_target))
    cls_target = tf.cast(cls_target, dtype=tf.int32)
    cls_target = tf.one_hot(cls_target, depth=2)

    # remove un-trained anchors from pred logit
    print(predict_logits,train_indices)
    logit0 = tf.gather_nd(predict_logits[..., 0], train_indices)
    logit1 = tf.gather_nd(predict_logits[..., 1], train_indices)
    logit = tf.stack([logit0, logit1], axis=1)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=cls_target, logits=logit)
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
    print(reg_target,pred_deltas)
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
    total_loss = fs_cls_loss + ss_cls_loss + fs_regr_loss + ss_regr_loss

    # fs_cls_loss = K.print_tensor(fs_cls_loss, message='fs_cls_loss = ')
    # ss_cls_loss = K.print_tensor(ss_cls_loss, message='ss_cls_loss = ')
    # fs_regr_loss = K.print_tensor(fs_regr_loss, message='fs_regr_loss = ')
    # ss_regr_loss = K.print_tensor(ss_regr_loss, message='ss_regr_loss = ')
    # total_loss = K.print_tensor(total_loss, message='total_loss = ')
    return total_loss
