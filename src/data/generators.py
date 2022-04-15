
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from ..analysis.plots import plot_seqs


class GeneratedSins (Dataset):

    def __init__(self, seq_len: int, N: int = None):
        self.len = N or 5000
        self.data = self.gen_sins(self.len, seq_len)
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        return self.data[i % self.len].unsqueeze(1)

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

    def __init__(self, seq_len: int, N: int = None):
        self.len = N or 5000
        self.data = self.gen_noise(self.len, seq_len)

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        return self.data[i % self.len].unsqueeze(1)

    def gen_noise(self, batch_size: int, seq_len: int):
        vertices = 6
        random = torch.rand(batch_size, vertices + 1)
        y = torch.zeros(batch_size, seq_len)

        def interp_eval(x):
            lo = int(x)
            t = x - lo
            return self.cerp(random[:, lo], random[:, lo + 1], t)

        for i in range(seq_len):
            x = i / float(seq_len + 1) * vertices
            y[:, i] = interp_eval(x)

        return y

    def cerp(self, a, b, x):
        g = (1 - np.cos(x * np.pi)) / 2
        return (1 - g) * a + g * b

    def lerp(self, a, b, x):
        return a * (1 - x) + b * x


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    # dataset = GeneratedSins(20)
    dataset = GeneratedNoise(20)
    dataloader = DataLoader(dataset, batch_size=8)

    x = next(iter(dataloader))
    print(x.shape)
