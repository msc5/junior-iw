
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
    return {tag: torch.tensor(values[tag]) for tag in tags}


def extract_data_from_files(glob_path, tags):
    losses = {key: [] for key in tags}
    labels = []
    min_length = 1e13
    for file in sorted(glob.glob(glob_path)):
        print(file)
        f = file.split(os.sep)
        labels += [f'{f[2][5:]} {f[3]} Layers']
        data = extract_data(file, tags)
        for t in tags:
            min_length = min(min_length, len(data[t]))
            losses[t] += [data[t]]
    print(min_length)
    tensors = {key: torch.stack([l[:min_length] for l in loss])
               for (key, loss) in losses.items()}
    tensors['labels'] = labels
    return tensors


if __name__ == "__main__":

    from rich import print

    LOSSES_PATH = 'src/analysis/losses/'

    # model = 'LSTM'
    model = 'ConvLSTM'

    test_tags = ['sequence/loss']
    train_tags = ['loss/train']

    train_glob_path = f'results/train/{model}*/*/*tfevents*'
    test_glob_path = f'results/test/{model}*/*/*tfevents*'

    # X = extract_data_from_files(test_glob_path, test_tags)
    # torch.save(X, f'{LOSSES_PATH}/{model}_seq_losses.pt')
    # Y = extract_data_from_files(train_glob_path, train_tags)
    # torch.save(Y, f'{LOSSES_PATH}/{model}_train_losses.pt')

    # # Plot Linear Datasets Train Loss
    # X = torch.load(f'{LOSSES_PATH}{model}_train_losses.pt')
    # print(X)
    # colors = plt.cm.winter(np.linspace(0, 1, len(X['loss/train'])))
    # fig = plt.figure(figsize=(12, 6))
    # plt.grid()
    # for i, x in enumerate(X['loss/train']):
    #     x_smooth = gaussian_filter1d(x[:30000], sigma=64)
    #     plt.plot(x_smooth, color=colors[i])
    # plt.legend([x[4:] for x in X['labels']], ncol=3)
    # # plt.legend(X['labels'], ncol=3)
    # plt.title(f'Smoothed {model} Training Loss')
    # plt.xlabel('Step')
    # plt.ylabel('MSE Loss')
    # plt.show()

    # Plot Test Seq Loss
    X = torch.load(f'{LOSSES_PATH}/{model}_seq_losses.pt')
    # colors = plt.cm.winter(np.linspace(0, 1, len(X['sequence/loss'])))
    # fig = plt.figure(figsize=(12, 6))
    # plt.grid()
    # for i, x in enumerate(X['sequence/loss']):
    #     plt.plot(x, color=colors[i])
    # plt.legend(X['labels'], ncol=3)
    # plt.title('MSE Loss over Predicted Sequence')
    # plt.xlabel('Predicted Sequence Step')
    # plt.ylabel('MSE Loss')
    # plt.show()

    # Average Test Losses
    for i, x in enumerate(X['sequence/loss']):
        avg = x.mean().item()
        label = X['labels'][i]
        print(f'{label:30} {avg:0.5e}')
