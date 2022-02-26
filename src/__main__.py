
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

import matplotlib.pyplot as plt

from arch.lstm import Seq2SeqLSTM

from rich import print


def gen_sin(T: int = 100):
    # Generates a sin wave with a random period (0, 2]
    # and a random phase (0, pi]. (All have length T)
    a = np.random.rand() * 4 * np.pi
    b = np.random.rand() * np.pi
    t = np.linspace(0, 1, T * 2)
    d = np.expand_dims((np.sin(a * t + b) + 1) / 2, 1)
    return d


def gen_data(T: int = 100, si: int = 10):
    data = np.concatenate([gen_sin(T) for _ in range(si)], axis=1)
    data = np.expand_dims(data, 0)
    return data[:, :T, :], data[:, T:, :]


def plot_output(x_train, y_train, output):
    print('Train: ', x_train.shape)
    print('Test: ', y_train.shape)
    print('Output: ', output.shape)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    t = np.linspace(0, 1, T * 2)
    for n in range(si):
        ax1.plot(t[:T], x_train[0, :, n], color='blue')
        ax1.plot(t[T:], y_train[0, :, n], color='blue')
        ax2.plot(t[:T], x_train[0, :, n], color='blue')
        ax2.plot(t[T:], y_train[0, :, n], color='limegreen')
        ax2.plot(t[T:], output[0, :, n], color='magenta')
    ax1.set_xlim([0, 1])
    ax2.set_xlim([0, 1])
    plt.show()


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

    plot_output(x_train, y_train, output.detach().cpu().numpy())

    x_train, y_train = gen_data(T, si)
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    for i in range(epochs):

        optimizer.zero_grad()

        #  x_train, y_train = gen_data(T, si)
        #  x_train = torch.from_numpy(x_train).float().to(device)
        #  y_train = torch.from_numpy(y_train).float().to(device)

        output = model(x_train)

        loss_it = loss_fn(output, y_train)
        loss_it.backward()
        loss = loss_it.item()

        optimizer.step()

        print(f'[{i:<4}]{"Loss:":>10}{loss:>10.5f}')

    plot_output(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        output.detach().cpu().numpy()
    )
