
import numpy as np
import matplotlib.pyplot as plt


def compare_prediction(output, truth):

    output_frames = output.squeeze().detach()[:, ...]
    truth_frames = truth.squeeze().detach()[:, ...]

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
