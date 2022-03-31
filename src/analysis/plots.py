
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":

    path = 'results2.npy'

    plot_loss(path)
