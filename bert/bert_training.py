import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import BertForSequenceClassification, AdamW
import utils.dataloading as dl
import utils.utils as utils

# Find GPU
device = torch.device("cuda")

# BERT constants:
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 16
pretrained_weights = 'bert-base-uncased'

# Set these paths to train BERT
MODEL_PATH = '../models/bert_mnli/bert_model_mnli.pt'
TRAIN_LOSS_PATH = '../models/bert_mnli/train_loss_per_batch.npy'
VAL_LOSS_PATH = '../models/bert_mnli/val_loss_per_epoch.npy'

# Initialize Model and Optimizer
model = BertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=3)
model.cuda()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)

# Load data
train_data = dl.SemEvalDataset("../data/preprocessed/bert_mnli_train.npy")
val_data = dl.SemEvalDataset("../data/preprocessed/bert_mnli_val.npy")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

# Track loss per batch
train_loss = []
val_loss_detailed = []
val_loss_per_epoch = []
tracker = 0
for i in trange(EPOCHS, desc="Epoch "):

    # Training
    model.train()
    training_loss = 0
    training_step = 0

    for step, batch in enumerate(train_loader):
        # Load batch on gpu
        batch = tuple(t.to(device) for t in batch)
        token_ids, segment, attention, lab = batch

        optimizer.zero_grad()
        outputs = model(token_ids, token_type_ids=segment, attention_mask=attention, labels=lab)
        loss = outputs[0]
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        training_step += 1

    print("Train Loss: {}".format(training_loss / training_step))

    # Validation
    model.eval()
    macro, weighted, acc = 0, 0, 0
    validation_step = 0
    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        token_ids, segment, attention, lab = batch
        with torch.no_grad():
            outputs = model(token_ids, token_type_ids=segment, attention_mask=attention, labels=lab)
        logits = outputs[1].detach().cpu().numpy()
        labels = lab.to('cpu').numpy()
        loss1 = utils.macro_f1(logits, labels)
        loss2 = utils.weighted_f1(logits, labels)
        loss3 = utils.accuracy(logits, labels)
        val_loss_detailed.append((loss1, loss2, loss3))
        macro += loss1
        weighted += loss2
        acc += loss3
        validation_step += 1
    macro = macro / validation_step
    weighted = weighted / validation_step
    acc = acc / validation_step
    val_loss_per_epoch.append((macro, weighted, acc))
    print("Macro-F1: ", macro)
    print("Weighted-F1: ", weighted)
    print("Accuracy: ", acc)
    new_tracker = (macro * 2 + weighted + acc) / 4
    if tracker < new_tracker:
        torch.save(model.state_dict(),
                   MODEL_PATH[:-3] + '_' + str(i) + '_' + str(new_tracker * 10000)[:4] + MODEL_PATH[-3:])
        tracker = new_tracker



np.save(TRAIN_LOSS_PATH, train_loss)
np.save(VAL_LOSS_PATH, val_loss_per_epoch)
