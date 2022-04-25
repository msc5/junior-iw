
import torch
import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d

import glob
from pathlib import Path
import os
import pandas as pd

from collections import defaultdict


def extract_data(path: Path, tags: str):
    dataset = tf.data.TFRecordDataset(path)
    values = defaultdict(list)
    for event in dataset:
        summary = event_pb2.Event.FromString(event.numpy()).summary.value
        for value in summary:
            values[value.tag] += [value.simple_value]
    return [values[tag] for tag in tags]


if __name__ == "__main__":

    from rich import print

    splits = ['train', 'test']
    # datasets = ['Stocks', 'GeneratedSins', 'GeneratedNoise']
    datasets = ['BAIR', 'KTH', 'MovingMNIST']
    layers = [3, 2, 1]

    test_tags = ['sequence/loss']
    train_tags = ['loss/train', 'loss/val']

    # # Extract Data
    # train_losses = []
    # val_losses = []
    # seq_losses = []
    # labels = []
    # glob_path = 'results/test/ConvLSTM*/*/*tfevents*'
    # # glob_path = 'results/test/LSTM*/*/*tfevents*'
    # print([f for f in glob.glob(glob_path)])
    # for file in glob.glob(glob_path):
    #     print(file)
    #     f = file.split(os.sep)
    #     labels += [f'{f[2][5:]} {f[3]} Layers']
    #     data = extract_data(file, test_tags)
    #     seq_losses += data
    #     # train_losses += [data[0][:54000]]
    #     # val_losses += [data[1]]
    #     # print(len(data[0]), len(data[1]))
    # print([len(x) for x in seq_losses])
    # # print([len(x) for x in train_losses])
    # # print([len(x) for x in val_losses])
    #
    # X = torch.tensor(seq_losses)
    # torch.save(X, 'ConvLSTM_seq_losses.pt')

    # X = torch.tensor(train_losses)
    # torch.save(X, 'ConvLSTM_train_losses.pt')
    #
    # Y = torch.tensor(val_losses)
    # torch.save(Y, 'ConvLSTM_val_losses.pt')

    labels = [f'{dataset} {layer} Layers'
              for dataset in datasets for layer in range(3, 0, -1)]
    print(labels)

    # Plot Linear Datasets Train Loss
    X = torch.load('ConvLSTM_train_losses.pt')
    colors = plt.cm.winter(np.linspace(0, 1, len(X)))
    fig = plt.figure(figsize=(12, 6))
    plt.grid()
    for i, x in enumerate(X):
        x_smooth = gaussian_filter1d(x[:30000], sigma=12)
        plt.plot(x_smooth, color=colors[i])
    plt.legend(labels, ncol=3)
    plt.title('Smoothed ConvLSTM Training Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.show()

    # # Plot Linear Datasets Test Seq Loss
    # X = torch.load('ConvLSTM_seq_losses.pt')
    # colors = plt.cm.winter(np.linspace(0, 1, len(X)))
    # fig = plt.figure(figsize=(12, 6))
    # plt.grid()
    # for i, x in enumerate(X):
    #     plt.plot(x, color=colors[i])
    # plt.legend(labels, ncol=3)
    # plt.title('MSE Loss over Predicted Sequence')
    # plt.xlabel('Predicted Sequence Step')
    # plt.ylabel('MSE Loss')
    # plt.show()
