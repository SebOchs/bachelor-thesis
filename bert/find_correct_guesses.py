import torch
from transformers import *
from bert import dataloading as dl
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer

# Find GPU
device = torch.device("cuda")

PATH = '../models/bert_asag/model.pt'
DATA = '../data/preprocessed/beetle_train.npy'

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Initialize Model and Optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(PATH))
model.cuda()
model.eval()

# Data to evaluate
beetle_data = dl.SemEvalDataset(DATA)
beetle_loader = DataLoader(beetle_data)
print(len(beetle_data))
correct_guesses = []
label = []
steps = []
with torch.no_grad():
    for step, batch in enumerate(beetle_loader):
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

data = np.array(list(zip(correct_guesses, label)))
np.save('../data/sear_data/correct_beetle', np.array(data), allow_pickle=True)



