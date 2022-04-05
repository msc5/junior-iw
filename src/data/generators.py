
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class GeneratedSins (Dataset):

    def __init__(self, seq_len: int, dat_size: int):
        self.len = 20000
        self.data = self.gen_sins(self.len, dat_size, seq_len)
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        return self.data[i % self.len]

    def gen_sin(self, seq_len: int):
        # Generates a sin wave with a random period (0, 2]
        # and a random phase (0, pi]. (All have length seq_len)
        a = np.random.rand() * 4 * np.pi
        b = np.random.rand() * np.pi
        t = np.linspace(0, 1, seq_len)
        d = (np.sin(a * t + b) + 1) / 2
        return d

    def gen_sins(
            self,
            batch_size: int,
            inp_size: int,
            seq_len: int,
    ):
        # Generates (inp_size) sin waves with random period and phase
        # with length seq_len
        data = np.stack([np.stack([self.gen_sin(seq_len)
                         for _ in range(inp_size)])
                         for _ in range(batch_size)])
        return data.transpose(0, 2, 1)


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
