# -*- coding: utf-8 -*-
"""
@project: 201904_dual_shot
@file: widerface.py
@author: danna.li
@time: 2019-06-06 16:45
@description:
"""
from keras.utils import Sequence
from dual_conf import current_config as conf
import os.path as osp
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

plt.switch_backend('agg')
WIDERFace_CLASSES = ['face']  # always index 0
WIDERFace_ROOT = conf.data_path


class WIDERFaceAnnotationTransform(object):
    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(WIDERFace_CLASSES, range(len(WIDERFace_CLASSES))))

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        for i in range(len(target)):
            target[i][0] = float(target[i][0]) / width
            target[i][1] = float(target[i][1]) / height
            target[i][2] = float(target[i][2]) / width
            target[i][3] = float(target[i][3]) / height
        return target


class WIDERFaceDetection(Sequence):
    """WIDERFace Detection Dataset Object
    http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDERFace folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'WIDERFace')
    """

    def __init__(self, root,
                 image_sets='train',
                 transform=None, target_transform=WIDERFaceAnnotationTransform(),
                 dataset_name='WIDER Face'):

        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.img_ids = list()
        self.label_ids = list()
        self.event_ids = list()
        if self.image_set == 'train':
            path_to_label = osp.join(self.root, 'wider_face_split')
            path_to_image = osp.join(self.root, 'WIDER_train/images')
            fname = "wider_face_train.mat"
        elif self.image_set == 'val':
            path_to_label = osp.join(self.root, 'wider_face_split')
            path_to_image = osp.join(self.root, 'WIDER_val/images')
            fname = "wider_face_val.mat"
        elif self.image_set == 'test':
            path_to_label = osp.join(self.root, 'wider_face_split')
            path_to_image = osp.join(self.root, 'WIDER_test/images')
            fname = "wider_face_test.mat"

        self.path_to_label = path_to_label
        self.path_to_image = path_to_image
        self.fname = fname
        self.f = scipy.io.loadmat(osp.join(self.path_to_label, self.fname))
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')
        self._load_widerface()

    def _load_widerface(self):
        error_bbox = 0
        train_bbox = 0
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                im_name = im[0][0]

                if self.image_set in ['test', 'val']:
                    self.img_ids.append(osp.join(self.path_to_image, directory, im_name + '.jpg'))
                    self.event_ids.append(directory)
                    self.label_ids.append([])
                    continue

                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                bboxes = []
                for i in range(face_bbx.shape[0]):
                    # filter bbox
                    if face_bbx[i][2] < 2 or face_bbx[i][3] < 2 or face_bbx[i][0] < 0 or face_bbx[i][1] < 0:
                        error_bbox += 1
                        # print (face_bbx[i])
                        continue
                    train_bbox += 1
                    xmin = float(face_bbx[i][0])
                    ymin = float(face_bbx[i][1])
                    xmax = float(face_bbx[i][2]) + xmin - 1
                    ymax = float(face_bbx[i][3]) + ymin - 1
                    bboxes.append([xmin, ymin, xmax, ymax])

                if (len(bboxes) == 0):  # filter bbox will make bbox none
                    continue
                self.img_ids.append(osp.join(self.path_to_image, directory, im_name + '.jpg'))
                self.event_ids.append(directory)
                self.label_ids.append(bboxes)
        self.label_ids = np.array(self.label_ids)
        # yield DATA(os.path.join(self.path_to_image, directory,  im_name + '.jpg'), bboxes)
        print("Error bbox number to filter : %d,  bbox number: %d" % (error_bbox, train_bbox))

    def __getitem__(self, index):
        im, gt, _ = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.img_ids)

    def pull_item(self, index):

        boxes = np.array(self.label_ids[index])
        img = cv2.imread(self.img_ids[index])
        labels = np.full(boxes.shape[0], 0)
        height, width, channels = img.shape
        if self.target_transform is not None:
            boxes = self.target_transform(boxes, width, height)
        # data augmentation
        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
        return img, boxes, labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    def plot_anchor(img_array, anchor_list):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img_array.astype(int))
        for a in anchor_list:
            x1, y1, x2, y2 = [int(i) for i in a]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()


    from prepare_data.augment import Augmentation

    to_aug = Augmentation(size=conf.net_in_size)
    dataset = WIDERFaceDetection(conf.data_path, 'train', to_aug, None)
    for i in range(10):
        image, gt, label = dataset.pull_item(i)
        plot_anchor(image, gt)
