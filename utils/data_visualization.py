import jsonlines
import numpy as np
import os
import xml.etree.ElementTree as et
from transformers import BertTokenizer
import matplotlib.pyplot as plt


def histograms():
    """
    Fast histogram of input length given a data set
    :return: None
    """
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True, do_basic_tokenize=True)
    path = "data/datasets/training/mnli/multinli_1.0_train.jsonl"

    answer_length = []
    file = jsonlines.open(path)
    for line in file:
        sent1 = line['sentence1']
        sent2 = line['sentence2']
        gold = line['gold_label']
        if gold not in ['entailment', 'contradiction', 'neutral']:
            continue
        len1 = len(tokenizer.tokenize(sent1))
        len2 = len(tokenizer.tokenize(sent2))

        answer_length.append(len1+len2+3)
    avg = sum(answer_length) / len(answer_length)
    answer_length = [x for x in answer_length if x < 200]
    plt.hist(answer_length, bins=range(min(answer_length), max(answer_length)))
    plt.title('Length distribution of premise and hypothesis')
    plt.xlabel('Total sequence length')
    plt.ylabel('Number of data instances')
    plt.savefig("mnli_trainingdatahist" + str(avg)[:5] +".png", dpi=600)

def label_dist(path):
    """
    Label distribution of a given data set
    :param path: data set location
    :return: None
    """
    data = np.load(path, allow_pickle=True)
    array = [x[3] for x in data]
    for i in set(array):
        print(i, " : ", sum([1 for x in array if x == i]))

#label_dist("data/preprocessed/bert_sciEntsBank_val.npy")
histograms()