
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

import matplotlib.pyplot as plt

from arch.lstm import Seq2SeqLSTM
from tests.sin_prediction_tests import gen_data, plot_sins

from rich import print

if __name__ == '__main__':

    #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    bs = 1
    T = 20
    si = 5
    sh = 30
    de = 1
    dd = 1
    model = Seq2SeqLSTM(bs, (si, sh), (de, dd)).to(device)

    x_train, y_train = gen_data(T, si)
    output = model(torch.from_numpy(x_train).to(device).float())

    epochs = 1000
    loss_fn = nn.MSELoss()
    #  loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    plot_sins(x_train, y_train, output.detach().cpu().numpy())

    #  x_train, y_train = gen_data(T, si)
    #  x_train = torch.from_numpy(x_train).float().to(device)
    #  y_train = torch.from_numpy(y_train).float().to(device)

    for i in range(epochs):

        optimizer.zero_grad()

        x_train, y_train = gen_data(T, si)
        x_train = torch.from_numpy(x_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)

        output = model(x_train)

        loss_it = loss_fn(output, y_train)
        loss_it.backward()
        loss = loss_it.item()

        optimizer.step()

        print(f'[{i:<4}]{"Loss:":>10}{loss:>10.5f}')

    plot_sins(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        output.detach().cpu().numpy()
    )
