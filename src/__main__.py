
from rich import print
# from tests.sequences import gen_sins, gen_lins, plot_seq
from .arch.lstm import Seq2SeqLSTM
from .arch.convlstm import ConvLSTMSeq2Seq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchvision.io import read_video

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .analysis.video import show_video, prediction_video


if __name__ == '__main__':

    # vid = read_video(
    #     'datasets/KTH/handclapping/person01_handclapping_d1_uncomp.avi')
    # print(vid)
    # print(vid[0].shape)
    #
    # print('Initializing Model...')
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # seq_len = 100
    # pred_len = 10
    # dat_size = 3
    # hid_size = 50
    # model_dep = 3
    #
    # model = Seq2SeqLSTM(dat_size, hid_size, model_dep).to(device)
    #
    # epochs = 50000
    # loss_fn = nn.MSELoss()
    # #  loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.005)
    #
    # x_train, y_train = gen_sins(dat_size, seq_len, pred_len)
    # x_train = torch.from_numpy(x_train).float().to(device)
    # y_train = torch.from_numpy(y_train).float().to(device)
    #
    # output = model(x_train, pred_len)
    #
    # fig, lines = plot_seq(
    #     x_train.detach().cpu().numpy(),
    #     y_train.detach().cpu().numpy(),
    #     output.detach().cpu().numpy()
    # )
    # plt.show()
    #
    # print('Training Model...')
    #
    # def forward(i):
    #     x_train, y_train = gen_lins(dat_size, seq_len, pred_len)
    #     x_train = torch.from_numpy(x_train).float().to(device)
    #     y_train = torch.from_numpy(y_train).float().to(device)
    #     optimizer.zero_grad()
    #     output = model(x_train, pred_len)
    #     loss_it = loss_fn(output[0], y_train[0])
    #     loss_it.backward()
    #     loss = loss_it.item()
    #     optimizer.step()
    #     print(f'[{i:<4}]{"Loss:":>10}{loss:>15.10f}')
    #     return output.detach().cpu().numpy()
    #
    # def animate(i):
    #     o = forward(i)
    #     for j in range(len(lines)):
    #         lines[j].set_ydata(o[0, :, j])
    #     return lines
    #
    # for i in range(epochs):
    #     output = forward(i)
    #
    # # ani = animation.FuncAnimation(
    # #     fig,
    # #     animate,
    # #     # frames=epochs,
    # #     interval=1,
    # # )
    # # plt.show()
    #
    # plot_seq(
    #     x_train.detach().cpu().numpy(),
    #     y_train.detach().cpu().numpy(),
    #     output
    # )
    # plt.show()

    pred_len = 10

    data = np.load('datasets/MovingMNIST/mnist_test_seq.npy')
    print('Loaded Data: ', data.shape)

    x_train = torch.tensor(data[:10, 0, :, :]).unsqueeze(
        0).unsqueeze(2).float()
    y_train = torch.tensor(data[10:, 0, :, :]).unsqueeze(
        0).unsqueeze(2).float()
    print(x_train.shape, y_train.shape)
    print(x_train.max(), x_train.min())

    model = ConvLSTMSeq2Seq(64, (1, 64, 64), 2)

    epochs = 500
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    output = model(x_train, 5)
    print(output.shape)

    prediction_video(x_train.squeeze(0), y_train.squeeze(0), output.squeeze(0))

    def forward(i):
        x_train = torch.tensor(data[:10, 0, :, :]).unsqueeze(
            0).unsqueeze(2).float()
        y_train = torch.tensor(data[10:, 0, :, :]).unsqueeze(
            0).unsqueeze(2).float()
        optimizer.zero_grad()
        output = model(x_train, pred_len) * 255
        loss_it = loss_fn(output[0], y_train[0])
        loss_it.backward()
        loss = loss_it.item()
        optimizer.step()
        print(f'[{i:<4}]{"Loss:":>10}{loss:>15.10f}')
        print(output.max(), output.min())
        return output.detach().cpu().numpy()

    for i in range(epochs):
        forward(i)

    prediction_video(x_train.squeeze(0), y_train.squeeze(0), output.squeeze(0))
