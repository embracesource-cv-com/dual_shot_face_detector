# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/26 
@file: augment.py
@description:
"""

import random
import cv2
import numpy as np
from dual_conf import current_config as conf
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000

def adjust_gt(bboxes, width, height, resized_width, resized_height):
    """ get the GT box coordinates, and resize to account for image resizing
    :param bboxes: 2-d array [[x1,y1,x2,y2]]
    :param width:
    :param height:
    :param resized_width:
    :param resized_height:
    :return:
    """
    num_bboxes = bboxes.shape[0]
    gta = np.zeros((num_bboxes, 4), dtype='int')
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = bbox[0] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox[1] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox[2] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox[3] * (resized_height / float(height))
    return gta


def aug_single_image(image, gts):
    """ crop sub_image making the size of the center face / size of sub_image = 640/rand(16,32,62,128,256,512)
    :param image: single image array in 3-d,eg.(w,h,c)
    :param gts: 2-d array,eg,(num_gts,x1,y1,x2,y2)
    :return: augmented image array and corresponding gts
    """
    anchor_scale = [16, 32, 56, 128, 512]
    rand_scale = random.choice(anchor_scale)
    face_ratio = rand_scale / conf.net_in_size

    # randomly select a gt
    rand_ind = np.random.randint(0, len(gts), 1)[0]
    gt = gts[rand_ind]
    w_gt = gt[2] - gt[0]
    h_gt = gt[3] - gt[1]
    x_c = w_gt / 2 + gt[0]
    y_c = h_gt / 2 + gt[1]

    # calculate sub_image area and crop it
    face_size = np.sqrt(w_gt * h_gt)
    sub_size = face_size / face_ratio
    sub_half = sub_size / 2
    sub_x1, sub_x2 = x_c - sub_half, x_c + sub_half
    sub_y1, sub_y2 = y_c - sub_half, y_c + sub_half
    pil_img = Image.fromarray(image)
    sub_img = np.array(pil_img.crop((sub_x1, sub_y1, sub_x2, sub_y2)))

    # transfer gt coordinates to fit the cropped image
    x_crop = gts[:, [0, 2]] - sub_x1
    y_crop = gts[:, [1, 3]] - sub_y1
    gt_crop = np.column_stack([x_crop[:, 0], y_crop[:, 0], x_crop[:, 1], y_crop[:, 1]])
    gt_crop = gt_crop.astype('int')

    # remove gt in corpped image which is less than 10 pixel
    gt_crop[gt_crop < 0] = 0
    gt_crop[gt_crop > sub_size] = sub_size
    gt_area = [(i[2] - i[0]) * (i[3] - i[1]) for i in gt_crop]
    big_gt_ind = [i for i, j in enumerate(gt_area) if j > 10]
    gt_in_crop = gt_crop[big_gt_ind]

    # resize to 640*640
    sub_img = cv2.resize(sub_img, dsize=(conf.net_in_size, conf.net_in_size),
                         interpolation=cv2.INTER_LINEAR)  # dsize(cols,rows)
    gt_in_crop = adjust_gt(gt_in_crop, sub_size, sub_size, conf.net_in_size, conf.net_in_size)
    return sub_img, gt_in_crop
