
import random
import torch
import torch.nn as nn
from pathlib import Path

from torch.utils.data import Dataset, DataLoader


class Stocks (Dataset):

    def __init__(self, seq_len: int = 20, split: str = 'train'):
        self.seq_len = seq_len
        self.split = split
        self.path = Path('src/data/datasets/stocks/raw_stocks')
        self.files = self.path.glob('*.pt')
        self.data = [torch.load(f) for f in self.files]
        self.lengths = [len(d) for d in self.data]
        self.len = sum([l // self.seq_len for l in self.lengths])
        self.slice = self.month = 0
        if self.split == 'test':
            self.month = 11
            self.len = self.lengths[11] // self.seq_len - 1

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        if self.split == 'train':
            if self.slice >= self.lengths[self.month] // self.seq_len:
                self.month += 1
                self.slice = 0
                if self.month >= 10:
                    raise StopIteration
            start = self.slice * self.seq_len
            end = start + self.seq_len
            self.slice += 1
            return self.data[self.month][start:end].unsqueeze(1)
        if self.split == 'test':
            if self.slice >= self.lengths[self.month] // self.seq_len:
                raise StopIteration
            self.slice += 1
            start = self.slice * self.seq_len
            end = start + self.seq_len
            return self.data[self.month][start:end].unsqueeze(1)


def make_dataset():

    import csv
    import time
    import pandas as pd

    from alpha_vantage.timeseries import TimeSeries
    ts = TimeSeries(key='A6YNKD8LYDFDEALD', output_format='csv')

    # for month in range():
    month = 11
    data, meta_data = ts.get_intraday_extended(
        'GOOGL', interval='1min', slice=f'year2month{month + 1}')
    x = [float(v[4]) for v in data if v[4] != 'close']
    x = torch.tensor(x)
    x = (x - x.std()) / x.mean()
    print(x)
    print(len(x))
    time.sleep(5)
    torch.save(x, f'stocks_year2month{month + 1}.pt')


if __name__ == "__main__":

    from rich import print

    ds = Stocks(seq_len=20, split='test')
    dl = DataLoader(ds, batch_size=4, drop_last=True)

    print(len(dl))
    for i, d in enumerate(dl):
        print(i, d.shape)
