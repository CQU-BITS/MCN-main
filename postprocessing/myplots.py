
# Description: Signal processing collaborated with deep learning aims to establish a novel causal network architecture and the proposed MCN achieved
# an interpretable expression with high accuracy. It should be noted that multiplication and convolution modules are mathematically connected as a 
# combination of BPNN and CNN, where the interpretable designed filters enhances the data mining ability.
# Authors   : Rui Liu, Xiaoxi Ding
# URL       : https://github.com/CQU-BITS/MCN-main
# The related reference : R. Liu, X. Ding*, Q. Wu, Q. He and Y. Shao, "An Interpretable Multiplication-Convolution Network for Equipment
# Intelligent Edge Diagnosis", IEEE Transactions on Systems, Man and Cybernetics: Systems,
# DOI: 10.1109/TSMC.2023.3346398
# Date     : 2024/01/16
# Version: v0.1.0
# Copyright by CQU-BITS

import pandas as pd
import numpy as np
import os
from pylab import *
from sklearn.manifold import TSNE
import itertools
from sklearn.metrics import confusion_matrix


plt.rcParams['figure.dpi'] = 600  # plt.show显示分辨率
plt.rcParams['axes.unicode_minus'] = False
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 16}
plt.rc('font', **font)


# 绘图颜色和标记符号
color = ['black', 'red', 'blue', 'green', 'cyan', 'magenta', 'darkkhaki', 'gray', 'blueviolet', 'olive', 'brown',
         'plum', 'maroon', 'yellow', 'salmon']
marker = ['o', 's', '^', 'p', 'X', '*', '8', 'D', '+', '<', '>', 'x', 'P', 'd', 'H']


def plotLossAcc(trainLoss, valLoss, trainAcc, valAcc, save_path, fig_name):

    epochs = np.arange(len(trainLoss))
    fig, subs = plt.subplots(2, figsize=(4, 7))  # 返回画布和子图

    subs[0].plot(epochs, trainLoss, 'g-', label='Training', lw=2)
    subs[0].plot(epochs, valLoss, 'r-', label='Validating', lw=2)
    subs[0].set_xlabel('Epoch')
    subs[0].set_ylabel('Loss')
    subs[0].legend()

    subs[1].plot(epochs, trainAcc, 'g-', label='Training', lw=2)
    subs[1].plot(epochs, valAcc, 'r-', label='Validating', lw=2)
    subs[1].set_xlabel('Epoch')
    subs[1].set_ylabel('Acc. (%)')
    subs[1].legend()
    plt.tight_layout()
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    plt.savefig(os.path.join(save_path, fig_name), dpi=600, bbox_inches='tight')
    plt.show()



def plotTSNECluster(fetures, labels, num_classes, typeLabel, save_path, fig_name):

    data, labels = fetures, labels
    data_tSNE = TSNE(n_components=2, init='pca', learning_rate=200, method='exact', random_state=0).fit_transform(data)

    for ii in range(num_classes):
        idx = np.where(np.array(labels) == ii)[0]
        exec('type' + str(ii) + ' = np.array(data_tSNE)[idx]')

    plt.figure(figsize=(4.5, 3))
    for jj in range(num_classes):
        plt.scatter(eval("type"+str(jj))[:, 0], eval("type"+str(jj))[:, 1], s=10, c=color[jj], marker=marker[jj])
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(labels=typeLabel, fontsize=13, loc='best')
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    plt.savefig(os.path.join(save_path, fig_name), dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plotConfusionMatrix(pred, labels, class_names, save_path, fig_name):

    cmtx = confusion_matrix(labels, pred)
    num_classes = len(class_names)

    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=(4, 4))
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(j, i, format(cmtx[i, j], "") if cmtx[i, j] != 0 else "0",
                 horizontalalignment="center",
                 verticalalignment='center', color=color)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    os.makedirs(save_path) if not os.path.exists(save_path) else None
    plt.savefig(os.path.join(save_path, fig_name), dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

