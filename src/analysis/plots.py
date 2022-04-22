
import io
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


def plot_seqs(x, y, output):
    def to_numpy(tensor): return tensor.squeeze().detach().cpu()
    x, y, output = [to_numpy(tensor) for tensor in [x, y, output]]
    batch_size, seq_len = x.shape
    _, fut_len = y.shape
    t = np.linspace(0, 1, seq_len + fut_len)
    seq_t, fut_t = t[:seq_len], t[seq_len:(seq_len + fut_len)]
    fig = plt.figure(figsize=(12, 6))
    plt.grid()
    for n in range(batch_size):
        plt.plot(seq_t, x[n], color='blue')
        plt.plot(fut_t, y[n], color='limegreen')
        plt.plot(fut_t, output[n], color='magenta')
    plt.title('Sequences and Predictions')
    return fig


def plot_loss_over_seq(losses):
    # batch_size, seq_len = y.shape[0], y.shape[1]
    # losses = [loss_fn(output[:, i], y[:, i]).item()
    #           for i in range(seq_len)]
    losses = losses.squeeze().detach().cpu()
    fig = plt.figure(figsize=(12, 6))
    plt.grid()
    plt.plot(losses)
    plt.title('Loss over Entire Sequence')
    return fig


def plot_to_tensor(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torch.nn import MSELoss
    from ..data.generators import GeneratedSins, GeneratedNoise

    from ..arch.lstm import LSTMSeq2Seq

    seq_len = 200

    dataset = GeneratedSins(seq_len)
    # dataset = GeneratedNoise(seq_len)
    dataloader = DataLoader(dataset, batch_size=8)

    data = next(iter(dataloader))
    x, y = data[:, :(seq_len // 2)], data[:, (seq_len // 2):]
    print(x.shape, y.shape)

    # fig = plot_seqs(x, y, y)
    # tensor = plot_to_tensor(fig)

    model = LSTMSeq2Seq(1, 64, 1)
    output = model(x)

    loss = MSELoss()
    fig = plot_loss_over_seq(loss, y, output)
    # tensor = plot_to_tensor(fig)

    plt.show()
