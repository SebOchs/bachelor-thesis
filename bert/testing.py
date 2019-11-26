import torch
from transformers import *

# Find GPU
device = torch.device("cuda")

PATH = 'out/model.pt'

# Initialize Model and Optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(PATH))
model.eval()
