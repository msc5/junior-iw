
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ..arch.convlstm import ConvLSTMSeq2Seq

INTERVAL = 150


def format_tensor(data: torch.Tensor):
    return data.detach().permute(0, 2, 3, 1)


def show_video(data: torch.Tensor):

    def update(i: int):
        frame.set_data(video[i])

    # --> (seq_len, img_chan, img_h, img_w)
    video = format_tensor(data)
    # --> (seq_len, img_h, img_w, img_chan)
    fig, ax = plt.subplots()
    frame = ax.imshow(video[0])
    ani = animation.FuncAnimation(
        fig, update, frames=len(video), interval=INTERVAL)
    plt.show()


def prediction_video(input_seq, true_seq, pred_seq):
    """
    Parameters
        input_seq: shape (seq_len, img_chan, img_h, img_w)
        true_seq: shape (pred_len, img_chan, img_h, img_w)
        pred_seq:  shape (pred_len, img_chan, img_h, img_w)
    """

    def update(i: int):
        true_frame.set_data(true[i])
        pred_frame.set_data(pred[i])
        return true_frame, pred_frame

    seq_len = input_seq.shape[0]
    pred_len = pred_seq.shape[0]
    total_len = seq_len + pred_len
    x = format_tensor(input_seq)
    y = format_tensor(true_seq)
    pred = format_tensor(pred_seq)

    fig, (true_ax, pred_ax) = plt.subplots(1, 2)
    true = np.concatenate((x, y))
    pred = np.concatenate((x, pred))
    true_frame = true_ax.imshow(true[0])
    pred_frame = pred_ax.imshow(pred[0])
    ani = animation.FuncAnimation(
        fig, update, interval=INTERVAL, frames=total_len)
    plt.show()


if __name__ == '__main__':

    data = np.load('datasets/MovingMNIST/mnist_test_seq.npy')
    print('Loaded Data: ', data.shape)

    x_train = torch.tensor(data[:10, 0, :, :]).unsqueeze(1)
    y_train = torch.tensor(data[10:, 0, :, :]).unsqueeze(1)
    print(x_train.shape, y_train.shape)
    print(x_train.max(), x_train.min())

    show_video(x_train)

    model = ConvLSTMSeq2Seq(64, (1, 64, 64), 3)

    output = model(x_train.unsqueeze(0).float(), 5)
    print(output.shape)

    prediction_video(x_train.squeeze(0), y_train, output.squeeze(0))
