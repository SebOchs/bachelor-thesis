# Various functions and methods for preprocessing/metric measurement/plotting etc...
import math
import os
import random
import xml.etree.ElementTree as et
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
# Preprocessing ########################################################################################################
########################################################################################################################

def label_to_int(lab):
    """
    Assignes label strings to a class
    :param lab: string
    :return: class number
    """
    if lab == 'incorrect':
        return 0
    if lab == 'contradictory':
        return 1
    if lab == 'correct':
        return 2
    else:
        raise ValueError


def token_seg_att(seq1, seq2, tokenizer, cls=['[CLS]'], sep=['[SEP]'], pad=['[PAD]'], max_tokens=128):
    """
    Attaches tokens to 2 sentences and transforms tokens to match BERT input
    :param tokenizer: Tokenizer to obtain token id's
    :param seq1: first sentence
    :param seq2: second sentence
    :param cls: CLS token, may be changed
    :param sep: SEP token, may be changed
    :param pad: PAD token, may be changed
    :param max_tokens: max sequence length
    :return: token id's of complete sequence, segmentation mask and attention mask
    """
    tok1 = tokenizer.tokenize(seq1)
    tok2 = tokenizer.tokenize(seq2)
    tokens = cls + tok1 + sep + tok2
    if len(tokens) > max_tokens - 1:
        tokens = tokens[:max_tokens - 1] + sep
    else:
        tokens = tokens + sep
    att_len = len(tokens)
    while len(tokens) < max_tokens:
        tokens = tokens + pad
    assert (len(tokens) == max_tokens)
    first_sep = tokens.index(sep[0])
    tok_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokens))
    seg = np.zeros(max_tokens)
    att = np.append(np.ones(att_len), np.zeros(max_tokens - att_len))
    if att_len < max_tokens:
        seg[first_sep + 1: att_len] = 1
    else:
        seg[first_sep + 1:] = 1
    return tok_ids, seg, att


def load_data(path, tokenizer):
    """
    Loads data from directory of XML files
    :param path: path to load data from
    :param tokenizer: Tokenizer for tokens
    :return: list of preprocessed data
    """
    array = []
    files = os.listdir(path)
    for file in files:
        root = et.parse(path + '/' + file).getroot()
        for ref_answer in root[1]:
            for stud_answer in root[2]:
                t, s, a = token_seg_att(ref_answer.text, stud_answer.text, tokenizer)
                label = label_to_int(stud_answer.get('accuracy'))
                array.append([t, s, a, label])
    return array


def create_npy(data, location, mode='test'):
    """
    Creates an npy file of preprocessed data
    :param data: list of preprocessed data
    :param location: where the file should be saved
    :param mode: if mode is train, we split the data for training and validation, else keep it together
    """
    if mode == 'train':
        assert (len(location) == 2)
        random.shuffle(data)
        split = math.floor(0.8 * len(data))
        np.save(location[0], data[:split], allow_pickle=True)
        np.save(location[1], data[split:], allow_pickle=True)
    elif mode == 'test':
        np.save(location, data, allow_pickle=True)

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
