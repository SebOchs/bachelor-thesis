import torch
from transformers import *
import utils.dataloading as dl
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer

# Find GPU
device = torch.device("cuda")

# Set three paths to find correct guesses of a BERT model:
# PATH: location of the model
# DATA: location of the data to predict
# CORRECT: location where to save correct guesses

PATH = '../models/bert_mnli/model_mnli.pt'
DATA = '../data/preprocessed/bert_mnli_mismatched_test.npy'
CORRECT = '../data/eval_data/bert_mnli_mm_correct'
testing = np.load(DATA, allow_pickle=True)
print(len(testing))
print(len([x for x in testing if x[3] == 0]))
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Initialize Model and Optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(PATH))
model.cuda()
model.eval()

# Data to evaluate
data = dl.SemEvalDataset(DATA)
loader = DataLoader(data)
print("Nr. of data instances: ", len(data))
correct_guesses = []
label = []
steps = []
with torch.no_grad():
    for step, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        token_ids, segment, attention, lab = batch
        outputs = model(token_ids, token_type_ids=segment, attention_mask=attention, labels=lab)
        logits = outputs[1].detach().cpu().numpy().squeeze()
        labels = lab.to('cpu').numpy()
        if labels[0] == np.argmax(logits):
            id_s = np.array(token_ids.squeeze().cpu())
            tokens = tokenizer.convert_ids_to_tokens(id_s)
            correct_guesses.append(tokens)
            label.append(labels[0])
print("Nr. of correct guesses: ", len(correct_guesses))
print("Nr. of correct incorrect predictions: ", len([x for x in label if x == 0]))
data = np.array(list(zip(correct_guesses, label)))
np.save(CORRECT, np.array(data), allow_pickle=True)



