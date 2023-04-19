
import sys
import numpy as np
import torch

np.set_printoptions(suppress=True, precision=4)
import numpy.random as r
import sklearn.metrics as m
import matplotlib.pyplot as plt


def ROC_AUC(LabelFlatten, OutputFlatten, ShowROC=False):
    fpr, tpr, th = m.roc_curve(LabelFlatten, OutputFlatten)
    AUC = m.auc(fpr, tpr)  # AUC其实就是ROC曲线下边的面积
    if ShowROC:
        plt.figure('ROC curve')
        plt.plot(fpr, tpr)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
    # plt.show()
    return fpr, tpr, AUC


def PRC_AP_MF(LabelFlatten, OutputFlatten, ShowPRC=False):
    precision, recall, th = m.precision_recall_curve(LabelFlatten, OutputFlatten)
    F1ScoreS = 2 * (precision * recall) / ((precision + recall) + sys.float_info.min)
    MF = F1ScoreS[np.argmax(F1ScoreS)]  # Maximum F-measure at optimal dataset scale
    AP = m.average_precision_score(LabelFlatten, OutputFlatten)  # AP其实就是PR曲线下边的面积
    if ShowPRC:
        plt.figure('Precision recall curve')
        plt.plot(recall, precision)
        plt.ylim([0.0, 1.0])
        plt.xlabel('recall')
        plt.ylabel('precision')
    # plt.show()
    return recall, precision, MF, AP


def iou_mean(pred, target, n_classes=1):
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes + 1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / n_classes
