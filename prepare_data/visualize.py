# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/28 
@file: visualize.py
@description:
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_anchor(img_array, anchor_list):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_array)
    for a in anchor_list:
        x1, y1, x2, y2 = [int(i) for i in a]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
