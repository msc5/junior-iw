
import os
import torch
import numpy as np

from torch.utils.data import Dataset


class GeneratedSins (Dataset):

    def __init__(self, seq_len: int):
        self.len = 20000
        self.data = self.gen_sins(self.len, seq_len * 2)
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

    def gen_sins(self, batch_size: int, seq_len: int,):
        # Generates (batch_size) sin waves with random period and phase
        # with length seq_len
        return np.stack([self.gen_sin(seq_len) for _ in range(batch_size)])


class GeneratedNoise (Dataset):

    def __init__(self, seq_len: int):
        self.len = 20000
        self.data = self.gen_sins(self.len, seq_len * 2)
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        return self.data[i % self.len]

    def gen_noise(self, seq_len: int):
        a, b = torch.rand(2)
        # x, y = 0,

    def cos_interpolate(a, b, x):
        f = (1 - np.cos(x * np.pi)) * 0.5
        return a + (1 - f) + b * f

    def lin_interpolate(a, b, x):
        return a * (1 - x) + b * x


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = GeneratedSins(20)
    dataloader = DataLoader(dataset, batch_size=8)

    x = next(iter(dataloader))
    print(x.shape)
