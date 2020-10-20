# -*- coding: utf-8 -*-
# Following sententes are meant to speed up arrays
# cython: boundscheck=False, wraparound=False, nonecheck=False
"""
Created on Thu Nov 29 10:18:43 2018

@author: gatv
"""
import sklearn.metrics as metrics
from tqdm import tqdm as progressBar
import numpy as np
# Methods to compute metrics from model


def getRecall(tp, fn):
    # calc recall
    return tp/(tp+fn)


def getSpecificity(tn, fp):
    # calc precision
    return tn/(tn+fp)


def getFPR(tn, fp):
    # false positive rate
    return fp/(fp+tn)


def getFNR(tp, fn):
    # calc precision
    return fn/(tp+fn)


def getPWC(tp, tn, fp, fn):
    # PWC
    return (100*(fp+fn))/(tp+tn+fp+fn)


def getPrecision(tp, fp):
    # precision
    return tp/(tp+fp)


def getFMeasure(precision, recall):
    # F-Measure
    return (2*precision*recall)/(precision+recall)


def computeMetrics(y_pred, y_true_generator, img_width, img_height):
    print('Initializating metrics measurement process ... ')
    y_pred_processed = np.zeros(y_pred.shape, dtype='uint8')
    print(y_true_generator.filenames[0])
    print(y_true_generator.filenames[(len(y_true_generator)-1)*32])
    length_y_pred = y_pred.shape[0]
    for pred_idx in range(0, length_y_pred):
        pred_array = y_pred[pred_idx]
        y_pred_value = pred_array*255
        y_pred_value = y_pred_value.astype("uint8")
        y_pred_processed[pred_idx] = y_pred_value
    y_true = np.ndarray([])
    labels = list(range(0, 256))
    # analyze each prediction, not all at once
    counter_samples = 0
    for batch_num in progressBar(range(0, len(y_true_generator))):
        # From 64x64x3 to 4096 vector
        y_true_value = y_true_generator[batch_num]*255
        y_true_value = y_true_value.astype("uint8")
        new_shape = (y_true_generator[batch_num].shape[0],
                     img_width*img_height)
        processed_batch = np.reshape(y_true_value, new_shape)
        if batch_num == 0:
            y_true = processed_batch
        else:
            y_true = np.concatenate((y_true, processed_batch), axis=0)
    current_recall = 0
    current_precision = 0
    fp, fn, tp, tn = 0, 0, 0, 0
    fp_px, fn_px, tp_px, tn_px = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred_processed):
        print('sample: {}'.format(counter_samples))
        confusion_matrix = metrics.confusion_matrix(true, pred, labels=labels)
        for px in labels:
            tp_px = confusion_matrix[px, px]
            fp_px = (sum(confusion_matrix[:, px]) - tp_px)
            fn_px = (sum(confusion_matrix[px, :]) - tp_px)
            tn_px = (confusion_matrix.sum() - (tp_px+fp_px+fn_px))
            tp += tp_px
            fp += fp_px
            fn += fn_px
            tn += tn_px
        print('FP:{} \n FN:{} \n TP:{} \n TN:{}'.format(fp, fn, tp, tn))

        current_recall = getRecall(tp, fn)
        current_specificity = getSpecificity(tn, fp)
        current_fpr = getFPR(tn, fp)
        current_fnr = getFNR(tp, fn)
        current_pwc = getPWC(tp, tn, fp, fn)
        current_precision = getPrecision(tp, fp)
        current_fmeasure = getFMeasure(current_precision, current_recall)
        print('current_recall: {}'.format(current_recall))
        print('current_specificity: {}'.format(current_specificity))
        print('current_fpr: {}'.format(current_fpr))
        print('current_fnr: {}'.format(current_fnr))
        print('current_pwc: {}'.format(current_pwc))
        print('current_precision: {}'.format(current_precision))
        print('current_fmeasure: {}'.format(current_fmeasure))
        counter_samples += 1
        