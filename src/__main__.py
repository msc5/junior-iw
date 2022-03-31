
from rich import print
# from tests.sequences import gen_sins, gen_lins, plot_seq
from .arch.lstm import Seq2SeqLSTM
from .arch.convlstm_ref_1 import EncoderDecoderConvLSTM as ConvLSTM_REF_1
from .arch.convlstm import ConvLSTMSeq2Seq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchvision.io import read_video

from .analysis.plots import compare_prediction

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred_len = 10

    data = np.load('datasets/MovingMNIST/mnist_test_seq.npy')
    print('Loaded Data: ', data.shape)

    def load(n: int, batch_size: int = 1):
        a = n % 10000
        b = (n + batch_size) % 10000
        x = torch.tensor(data[:10, a:b, :, :]).unsqueeze(2).float() / 255
        x = x.permute(1, 0, 2, 3, 4)
        y = torch.tensor(data[10:, a:b, :, :]).unsqueeze(2).float() / 255
        y = y.permute(1, 0, 2, 3, 4)
        # --> (batch_size, seq_len, img_chan, img_h, img_w)
        return x.to(device), y.to(device)

    def range_tensor(x):
        return (x.min().item(), x.max().item())

    n_vid = 55
    batch_size = 20
    seq_len = 10
    x_train, y_train = load(n_vid, batch_size)

    model = ConvLSTMSeq2Seq(64, (1, 64, 64), 2, batch_size).to(device)
    # model = ConvLSTM_REF_1(64, 1).to(device)

    epochs = 5000
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    output = model(x_train, seq_len)
    print(output.shape)

    compare_prediction(output, y_train)

    results = np.zeros((epochs, 3))

    def forward(i):
        x_train, y_train = load(i, batch_size)
        optimizer.zero_grad()
        output = model(x_train, seq_len)
        # compare_prediction(output, y_train)
        loss_it = loss_fn(output.squeeze(), y_train.squeeze())
        loss_it.backward()
        loss = loss_it.item()
        optimizer.step()
        out_min, out_max = range_tensor(output)
        true_min, true_max = range_tensor(y_train)
        print((f'[{i:<4}]'
               f'{"Loss:":>10}'
               f'{loss:>15.10f}'
               f'{"":>5}({out_min:<7.5f},{out_max:>8.5f})'))
        results[i, :] = np.array([loss, out_min, out_max])

    for i in range(epochs):
        forward(i)

    np.save('results3', results)

    output = model(x_train, seq_len)

    print(range_tensor(output))
    print(range_tensor(y_train))
    compare_prediction(output, y_train)
