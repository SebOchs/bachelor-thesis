import re
import torch

# Example finding for thesis

device = torch.device("cuda")
from bert.preprocessing_bert import BertPreprocessor
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import torch.nn.functional as F

pretrained_weights = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
PATH = '../models/bert_sciEntsBank/model.pt'
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(PATH))
model.cuda()
model.eval()


def list_to_string(list):
    """
    joins a list of strings together
    :param list: list of strings
    :return: string
    """
    return ' '.join(list)


def separate_answers(bert_text, cls='[CLS]', sep='[SEP]'):
    """
    Separates the sentences of sequence classification used for bert
    :param bert_text: list of bert word tokens
    :param cls: string of cls token
    :param sep: string of sep token
    :return: separated strings
    """
    # Fix double-hash
    pattern = '^##.*'
    remove = []
    for i, word in enumerate(bert_text):
        if re.match(pattern, word):
            bert_text[i] = bert_text[i - 1] + word[2:]
            remove.append(i - 1)
    for j in sorted(remove, reverse=True):
        bert_text.pop(j)
    cls_idx = bert_text.index(cls)
    sep_1_idx = bert_text.index(sep)
    ans1 = bert_text[cls_idx + 1:sep_1_idx]
    ans2 = bert_text[sep_1_idx + 1:bert_text.index(sep, sep_1_idx + 1)]
    return ans1, ans2


def predict(model, ref, stud, orig_pred):
    if type(ref) is list:
        ref = list_to_string(ref)
    if type(stud) is list:
        stud = list_to_string(stud)
    assert type(stud) is str and type(ref) is str
    token_ids, segment, attention, lab = \
        BertPreprocessor(bert_tokenizer, data=[ref, stud, orig_pred]).load_data()
    token_ids = torch.tensor([token_ids]).long().to(device)
    segment = torch.tensor([segment]).long().to(device)
    attention = torch.tensor([attention]).long().to(device)
    outputs = model.forward(input_ids=token_ids, attention_mask=attention, token_type_ids=segment)
    logits = outputs[0].detach().cpu().squeeze()

    return logits

a = predict(model, "if the motor runs , the object is a conductor .",
            "he will know because a conductor inevitably is not glowing in a circuit .", 0)

print(int(np.argmax(a)), F.softmax(a)[int(np.argmax(a))])
