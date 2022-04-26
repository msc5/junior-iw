import os
import io
import numpy as np
from PIL import Image
import torch

from torchvision.transforms import ToTensor


class BAIR (object):

    """Data Handler that loads robot pushing data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root
        if train:
            self.data_dir = '%s/processed_data/train' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s/processed_data/test' % self.root_dir
            self.ordered = True
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading
        self.d = 0
        self.totensor = ToTensor()

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return len(self.dirs)

    def get_seq(self):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d += 1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]
        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            # im = imread(fname).reshape(1, 64, 64, 3)
            # im = np.array(Image.open(fname)).reshape((1, 3, 64, 64))
            im = self.totensor(Image.open(fname)).reshape(1, 3, 64, 64)
            image_seq.append(im)
        image_seq = torch.cat(image_seq, axis=0)
        return image_seq

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_dataset = BAIR('src/data/datasets/BAIR/raw', train=True)
    train_dataloader = DataLoader(train_dataloader, batch_size=4)
    print(len(train_dataset, train_dataloader))
