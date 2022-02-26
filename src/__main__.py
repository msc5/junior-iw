
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from arch.lstm import Seq2SeqLSTM
from tests.sequences import gen_sins, gen_lins, plot_seq

from rich import print

if __name__ == '__main__':

    #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    bs = 1
    sl = 50
    fl = 50
    si = 10
    sh = 50
    de = 1
    dd = 1
    model = Seq2SeqLSTM(bs, (si, sh), (de, dd)).to(device)

    epochs = 1000
    loss_fn = nn.MSELoss()
    #  loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    x_train, y_train = gen_lins(si, sl, fl)
    #  x_train, y_train = gen_sins(si, sl, fl)
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    output = model(x_train, fl)

    fig, lines = plot_seq(x_train, y_train, output.detach().cpu().numpy())
    plt.show()

    def forward(i):
        #  x_train, y_train = gen_lins(si, sl, fl)
        #  x_train = torch.from_numpy(x_train).float().to(device)
        #  y_train = torch.from_numpy(y_train).float().to(device)
        optimizer.zero_grad()
        output = model(x_train, fl)
        loss_it = loss_fn(output[0], y_train[0])
        loss_it.backward()
        loss = loss_it.item()
        optimizer.step()
        print(f'[{i:<4}]{"Loss:":>10}{loss:>15.10f}')
        return output.detach().cpu().numpy()

    #  def animate(i):
    #      o = forward(i)
    #      for j in range(len(lines)):
    #          lines[j].set_ydata(o[0, :, j])
    #      return lines

    for i in range(epochs):
        output = forward(i)

    #  ani = animation.FuncAnimation(
    #      fig,
    #      animate,
    #      frames=epochs,
    #      interval=1,
    #  )
    #  plt.show()

    plot_seq(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        output
    )
    plt.show()
