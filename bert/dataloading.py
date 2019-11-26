import numpy as np
import torch
from torch.utils.data import Dataset


class SemEvalDataset(Dataset):

    def __init__(self, filename):
        self.data = np.load(filename, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ids, seg, att, lab = self.data[index]
        return torch.tensor(ids).long(), torch.tensor(seg).long(), torch.tensor(att).long(), lab
