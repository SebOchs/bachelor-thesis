import torch
import utils.utils as utils
from transformers import BertForSequenceClassification, BertTokenizer
import utils.dataloading as dl
from torch.utils.data import DataLoader
import numpy as np

# Find GPU
device = torch.device("cuda")

# Set three paths to test a BERT model:
# PATH: model location
# DATA: preprocessed test set
# LOSS: location of the loss file

PATH = '../models/bert_mnli/model_mnli.pt'
DATA = '../data/preprocessed/bert_mnli_mismatched_test.npy'
LOSS = '../models/bert_mnli/test_matched_loss'

# Initialize Model and Optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(PATH))
model.cuda()
model.eval()

# Data to evaluate
test_data = dl.SemEvalDataset(DATA)
test_loader = DataLoader(test_data)


data = []
logit_list = np.empty([1,3], dtype=float)
label_list = np.empty(1, dtype=int)

with torch.no_grad():
    macro, weighted, acc = 0, 0, 0

    for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        token_ids, segment, attention, lab = batch
        with torch.no_grad():
            outputs = model(token_ids, token_type_ids=segment, attention_mask=attention, labels=lab)
        logits = outputs[1].detach().cpu().numpy()
        labels = lab.to('cpu').numpy()
        logit_list = np.concatenate((logit_list, logits))
        label_list = np.concatenate((label_list, labels))

    loss1 = utils.macro_f1(logit_list, label_list)
    loss2 = utils.weighted_f1(logit_list, label_list)
    loss3 = utils.accuracy(logit_list, label_list)

    print("Macro-F1: ", loss1)
    print("Weighted-F1: ", loss2)
    print("Accuracy: ", loss3)
    data.append([loss1, loss2, loss3])


np.save(LOSS, np.array(data), allow_pickle=True)



