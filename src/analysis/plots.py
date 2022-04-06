
import io
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


def plot_loss(path):
    loss = np.load(path)
    fig = plt.figure()
    plt.plot(loss)
    plt.legend(['Loss', 'Output Minimum', 'Output Maximum'])
    plt.show()


def compare_prediction(output, truth):

    output_frames = output.detach().cpu()[0, :, ...]
    truth_frames = truth.detach().cpu()[0, :, ...]

    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 10, figsize=(18, 4))

    # Plot the original frames.
    for i, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(truth_frames[i]), cmap="gray")
        ax.set_title(f"Frame {i + 11}")
        ax.axis("off")

    # Plot the new frames.
    for i, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(output_frames[i]), cmap="gray")
        ax.set_title(f"Frame {i + 11}")
        ax.axis("off")

    # Display the figure.
    plt.show()


def plot_sins(x, y, output):
    assert x.shape == y.shape
    assert x.shape == output.shape
    def to_numpy(tensor): return tensor.squeeze().detach()
    x, y, output = [to_numpy(tensor) for tensor in [x, y, output]]
    batch_size, seq_len = x.shape
    _, fut_len = y.shape
    t = np.linspace(0, 1, seq_len + fut_len)
    seq_t, fut_t = t[:seq_len], t[seq_len:(seq_len + fut_len)]
    fig = plt.figure(figsize=(12, 6))
    for n in range(batch_size):
        plt.plot(seq_t, x[n], color='blue')
        plt.plot(fut_t, y[n], color='limegreen')
        plt.plot(fut_t, output[n], color='magenta')
    plt.title('Sin Wave Sequences and Predictions')
    return fig


def plot_to_tensor(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    return transforms.ToTensor()(image)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from ..data.generators import GeneratedSins

    seq_len = 20

    dataset = GeneratedSins(seq_len)
    dataloader = DataLoader(dataset, batch_size=8)

    data = next(iter(dataloader))
    x, y = data[:, :seq_len], data[:, seq_len:]

    fig = plot_sins(x, y, y)
    tensor = plot_to_buf(fig)

    plt.show()
