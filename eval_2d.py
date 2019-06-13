# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: eval_2d.py
@author: danna.li
@time: 2019-06-13 15:39
@description: 
"""
import numpy as np
import os
from prepare_data.model_target import iou_np
from dual_conf import current_config as conf
import glob


def filter_by_score(pred_boxes, pred_labels, pred_scores, score_thread):
    masks = [i > score_thread for i in pred_scores]
    pred_scores = [pred_scores[mask] for pred_scores, mask in zip(pred_scores, masks)]
    pred_labels = [pred_labels[mask] for pred_labels, mask in zip(pred_labels, masks)]
    pred_boxes = [pred_boxes[mask] for pred_boxes, mask in zip(pred_boxes, masks)]
    return pred_boxes, pred_labels, pred_scores


def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [pred_labels[mask] for pred_labels, mask in zip(pred_labels, score_seq)]
    pred_scores = [pred_scores[mask] for pred_scores, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(y1,x1,y2,x2)),(b,(y1,x1,y2,x2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(y1,x1,y2,x2)),(n,(y1,x1,y2,x2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 3, 0 for background,only calculate cls index in range(num_cls)[1:]
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                iou = iou_np(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)  # score大的index在前
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
        print('recall:', recall)
        print('precision:', precision)
    return all_ap


def load_predicted_file(saved_file_path):
    files = glob.glob(os.path.join(saved_file_path, '*.npz'))
    files.sort()
    gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores = [], [], [], [], []
    for file in files:
        with np.load(file)as data:
            gt_boxes.append(data['gt_boxes'])
            gt_labels.append(data['gt_labels'])
            pred_boxes.append(data['pred_boxes'])
            pred_labels.append(data['pred_labels'])
            pred_scores.append(data['pred_scores'])
    return gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores


def main():
    iou_thread = 0.5
    score_thread = 0.8
    saved_file_path = os.path.join(conf.output_dir, 'widerface_val_result')
    gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores = load_predicted_file(saved_file_path)
    print(len(pred_boxes), pred_boxes[0].shape, pred_labels[0].shape, pred_scores[0].shape)
    pred_boxes, pred_labels, pred_scores = filter_by_score(pred_boxes, pred_labels, pred_scores, score_thread)
    print(len(pred_boxes), pred_boxes[0].shape, pred_labels[0].shape, pred_scores[0].shape)
    pred_boxes, pred_labels, pred_scores = sort_by_score(pred_boxes, pred_labels, pred_scores)
    all_ap = eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, conf.num_class)
    print('AP for all cls:', all_ap)
    return all_ap


if __name__ == '__main__':
    main()
