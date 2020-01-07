# Various functions and methods for preprocessing/metric measurement/plotting etc...

import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# Metrics ##############################################################################################################
########################################################################################################################


def f1(pr, tr, class_num):
    """
    Calculates F1 score for a given class
    :param pr: list of predicted values
    :param tr: list of actual values
    :param class_num: indicates class
    :return: f1 score of class_num for predicted and true values in pr, tr
    """

    # Filter lists by class
    pred = [x == class_num for x in pr]
    truth = [x == class_num for x in tr]
    mix = list(zip(pred, truth))
    # Find true positives, false positives and false negatives
    tp = mix.count((True, True))
    fp = mix.count((False, True))
    fn = mix.count((True, False))
    # Return f1 score, if conditions are met
    if tp == 0 and fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp == 0 and fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if recall == 0 and precision == 0:
        return 0
    else:
        return 2 * recall * precision / (recall + precision)


def macro_f1(predictions, truth):
    """
    Calculates macro f1 score, where all classes have the same weight
    :param predictions: logits of model predictions
    :param truth: list of actual values
    :return: macro f1 between model predictions and actual values
    """
    flatten_pred = np.argmax(predictions, axis=1).flatten()
    labels_flat = truth.flatten()
    f1_0 = f1(flatten_pred, labels_flat, 0)
    f1_1 = f1(flatten_pred, labels_flat, 1)
    f1_2 = f1(flatten_pred, labels_flat, 2)
    if np.sum([x == 1 for x in labels_flat]) == 0:
        return (f1_0 + f1_2) / 2
    else:
        return (f1_0 + f1_1 + f1_2) / 3


def weighted_f1(predictions, truth):
    """
    Calculates weighted f1 score, where all classes have different weights based on appearance
    :param predictions: logits of model predictions
    :param truth: list of actual values
    :return: weighted f1 between model predictions and actual values
    """
    flatten_pred = np.argmax(predictions, axis=1).flatten()
    labels_flat = truth.flatten()
    weight_0 = np.sum([x == 0 for x in truth])
    weight_1 = np.sum([x == 1 for x in truth])
    weight_2 = np.sum([x == 2 for x in truth])
    f1_0 = f1(flatten_pred, labels_flat, 0)
    f1_1 = f1(flatten_pred, labels_flat, 1)
    f1_2 = f1(flatten_pred, labels_flat, 2)
    return (weight_0 * f1_0 + weight_1 * f1_1 + weight_2 * f1_2) / len(truth)


def accuracy(predictions, truth):
    """
    Calculates flat accuracy
    :param predictions:
    :param truth:
    :return: accuracy
    """
    flatten_pred = np.argmax(predictions, axis=1).flatten()
    labels_flat = truth.flatten()
    return np.sum(flatten_pred == labels_flat) / len(truth)

########################################################################################################################
# Plotting #############################################################################################################
########################################################################################################################

def plot(path, location, metric_names):
    """
    Creates an image that displays metrics etc
    :param path: data location
    :param location: image name
    :param axis_name: list of metric names
    """
    y = np.load(path)
    metric = len(y.shape)

    if metric == 1 and metric_names[0] == 'Train Loss':
        plotting_list = []
        x = y.shape[0] / 8
        for i in range(8):
            plotting_list.append(np.sum(y[int(i*x):int((i+1)*x)]))
        plt.plot(plotting_list)
        plt.xlabel("Epoch")
        plt.ylabel(metric_names[0])
    elif metric > 1:
        for i in range(y.shape[1]):
            plt.plot(y[:, i], label=metric_names[i])
        plt.legend(loc='lower right')
    plt.savefig(location)

    return 0

########################################################################################################################
# Transformations ######################################################################################################
########################################################################################################################

