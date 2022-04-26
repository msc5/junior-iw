
import random
import torch
import torch.nn as nn
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

RAW_PATH = 'src/data/datasets/Stocks/raw'
APIKEY = 'A6YNKD8LYDFDEALD'


class Stocks (Dataset):

    def __init__(self, seq_len: int = 20, split: str = 'train'):
        self.seq_len = seq_len
        self.split = split
        self.path = Path(RAW_PATH)
        if self.split == 'train':
            self.files = self.path.glob('[!TSLA]*')
        elif self.split == 'test':
            self.files = self.path.glob('TSLA*')
        self.data = [torch.load(f) for f in self.files]
        self.lengths = [len(d) for d in self.data]
        self.len = sum([l // self.seq_len for l in self.lengths])
        self.buckets = {}
        count = 0
        for i, l in enumerate(self.lengths):
            for _ in range(l // self.seq_len):
                self.buckets[count] = i
                count += 1

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        file = self.buckets[i]
        prior = sum([l // self.seq_len for l in self.lengths[:file]])
        start = (i - prior) * self.seq_len
        end = start + self.seq_len
        slice = self.data[file][start:end].unsqueeze(1)
        # Normalize Data
        slice -= slice.min()
        slice /= slice.max()
        return slice


def make_dataset():

    import csv
    import time
    import os
    import pandas as pd
    from alpha_vantage.timeseries import TimeSeries

    # symbols = ['GOOGL', 'MSFT', 'TSLA', 'AAPL',
    #            'AMZN', 'NVDA', 'FB', 'AMD']

    symbols = ['GOOGL', 'MSFT', 'TSLA', 'AAPL',
               'AMZN', 'NVDA', 'FB', 'AMD',
               'BABA', 'PYPL', 'CRM', 'ATVI',
               'EA', 'IBM', 'ASML', 'INTC']

    ts = TimeSeries(key=APIKEY, output_format='csv')

    def retry_download(year, month, symbol, slice):
        print((f'Downloading {symbol:10} '
               f'year {year} month {month}\n'
               f'Slice {slice}'))
        data, meta_data = ts.get_intraday_extended(
            symbol=symbol, interval='1min', slice=slice)
        data = [d for d in data]
        if data:
            x = [float(v[4]) for v in data if v[4] != 'close']
            x = torch.tensor(x)
        else:
            print('Retrying...')
            return retry_download(year, month, symbol, slice)
        print('Download Successful:')
        print(len(x))
        torch.save(x, path)
        time.sleep(20)
        return x

    for year in [1, 2]:
        for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            for symbol in symbols:
                slice = f'year{year}month{month}'
                path = f'{RAW_PATH}/{symbol}_{slice}.pt'
                if not os.path.exists(path):
                    retry_download(year, month, symbol, slice)
                else:
                    pass
                    # print((f'Already Downloaded {symbol:10} '
                    #        f'year {year} month {month}'))
    print('Dataset Downloaded Successfully!')


if __name__ == "__main__":

    from rich import print

    make_dataset()

    # seq_len, batch_size = 50, 256
    seq_len, batch_size = 100, 256

    train_ds = Stocks(seq_len=seq_len, split='train')
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, drop_last=True, shuffle=True)
    test_ds = Stocks(seq_len=seq_len, split='test')
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, drop_last=True, shuffle=True)

    print(len(train_ds))
    print(len(test_ds))
    print(len(train_dl))
    print(len(test_dl))

    # for i, d in enumerate(train_dl):
    #     print(i, d.shape)
    # for i, d in enumerate(test_dl):
    #     print(i, d.shape)
