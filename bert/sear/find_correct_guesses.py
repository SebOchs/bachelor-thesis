import torch
from transformers import *
from bert import dataloading as dl
from torch.utils.data import DataLoader
import numpy as np
# Find GPU
device = torch.device("cuda")

PATH = '../out/model.pt'
DATA = '../../data/beetle_training.npy'

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
with torch.no_grad():
    for step, batch in enumerate(beetle_loader):
        batch = tuple(t.to(device) for t in batch)
        token_ids, segment, attention, lab = batch
        outputs = model(token_ids, token_type_ids=segment, attention_mask=attention, labels=lab)
        logits = outputs[1].detach().cpu().numpy()
        labels = lab.to('cpu').numpy()
        if labels[0] == np.argmax(logits):
            correct_guesses.append(token_ids)
            label.append(labels[0])

        print(step)
np.save('../../data/correct_guesses_beetle.npy', np.array(correct_guesses), allow_pickle=True)
np.save('../../data/labels_beetle.npy', np.array(label), allow_pickle=True)


