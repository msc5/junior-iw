
import os
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussian

from torch.utils.data import Dataset

from ..analysis.plots import plot_seqs

# DEFAULT_LEN = 55000
DEFAULT_LEN = 22000


class GeneratedSins (Dataset):

    def __init__(self, seq_len: int, N: int = None):
        self.len = N or DEFAULT_LEN
        self.seq_len = seq_len
        path = f'src/data/datasets/GeneratedSins/raw/{self.seq_len}_{self.len}.pt'
        if os.path.exists(path):
            print('Loading GeneratedSins...')
            self.data = torch.load(path)
        else:
            print('Making GeneratedSins...')
            self.data = self.gen_sins(self.len, seq_len)
            self.data = torch.from_numpy(self.data).float()
            torch.save(self.data, path)

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
        self.len = N or DEFAULT_LEN
        self.seq_len = seq_len
        path = f'src/data/datasets/GeneratedNoise/raw/{self.seq_len}_{self.len}.pt'
        if os.path.exists(path):
            print('Loading GeneratedNoise...')
            self.data = torch.load(path)
        else:
            print('Making GeneratedNoise...')
            self.data = self.gen_noise(self.len, seq_len)
            torch.save(self.data, path)

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        return self.data[i % self.len].unsqueeze(1)

    def gen_noise(self, batch_size: int, seq_len: int):
        def make_data():
            x = gaussian(np.random.rand(seq_len), sigma=1)
            x = torch.from_numpy(x).float()
            x -= x.min()
            x /= x.max()
            return x
        data = [make_data() for _ in range(batch_size)]
        return torch.stack(data)


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    sins = GeneratedSins(50)
    noise = GeneratedNoise(50)

    sins_dl = DataLoader(
        sins, batch_size=64, drop_last=True, shuffle=True)
    noise_dl = DataLoader(
        noise, batch_size=64, drop_last=True, shuffle=True)

    print(len(sins))
    print(len(noise))
    print(len(sins_dl))
    print(len(noise_dl))

    # dataset = GeneratedSins(20)
    # dataset = GeneratedNoise(20)
    # dataloader = DataLoader(dataset, batch_size=8)
    #
    # x = next(iter(dataloader))
    # print(x.shape)
