
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MovingMNIST (Dataset):

    def __init__(
            self,
            path,
            train: bool = True
    ):
        self.path = path
        self.data = torch.from_numpy(np.load(self.path))
        self.data = self.data.permute(1, 0, 2, 3).float()
        self.len = 10000

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        return self.data[i % self.len].unsqueeze(1)
