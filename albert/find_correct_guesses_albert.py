import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer
import utils.dataloading as dl
from torch.utils.data import DataLoader
import numpy as np

# Set three paths to find correct guesses of an ALBERT model:
# PATH: location of the model
# DATA: location of the data to predict
# CORRECT: location where to save correct guesses

# Find GPU
device = torch.device("cuda")

PATH = '../models/albert_beetle/albert_model.pt'
DATA = '../data/preprocessed/albert_beetle_uq_test.npy'
CORRECT = '../data/eval_data/albert_bee_uq_test_correct'

pretrained_weights = 'albert-base-v1'
tokenizer = AlbertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True, do_basic_tokenize=True)

# Initialize Model and Optimizer
model = AlbertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=3)
model.load_state_dict(torch.load(PATH))
model.cuda()
model.eval()

# Data to evaluate
beetle_data = dl.SemEvalDataset(DATA)
beetle_loader = DataLoader(beetle_data)
print("Nr. of data instances: ", len(beetle_data))
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
            text = tokenizer.convert_tokens_to_string(tokens)
            correct_guesses.append(text)
            label.append(labels[0])
print("Nr. of correct guesses: ", len(correct_guesses))
print("Nr. of correct incorrect predictions: ", len([x for x in label if x == 0]))
data = np.array(list(zip(correct_guesses, label)))
np.save(CORRECT, np.array(data), allow_pickle=True)



