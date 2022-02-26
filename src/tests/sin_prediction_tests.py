
import numpy as np
import matplotlib.pyplot as plt


def gen_sin(T: int = 100):
    # Generates a sin wave with a random period (0, 2]
    # and a random phase (0, pi]. (All have length T)
    a = np.random.rand() * 4 * np.pi
    b = np.random.rand() * np.pi
    t = np.linspace(0, 1, T * 2)
    d = np.expand_dims((np.sin(a * t + b) + 1) / 2, 1)
    return d


def gen_data(T: int = 100, si: int = 10):
    data = np.concatenate([gen_sin(T) for _ in range(si)], axis=1)
    data = np.expand_dims(data, 0)
    return data[:, :T, :], data[:, T:, :]


def plot_sins(x_train, y_train, output):
    print('Train: ', x_train.shape)
    print('Test: ', y_train.shape)
    print('Output: ', output.shape)
    assert y_train.shape == output.shape
    _, sl, si = x_train.shape
    fl = y_train.shape[1]
    t = np.linspace(0, 1, sl * 2)
    st, ft = t[:sl], t[sl:sl + fl]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    for n in range(si):
        ax1.plot(st, x_train[0, :, n], color='blue')
        ax1.plot(ft, y_train[0, :, n], color='blue')
        ax2.plot(st, x_train[0, :, n], color='blue')
        ax2.plot(ft, y_train[0, :, n], color='limegreen')
        ax2.plot(ft, output[0, :, n], color='magenta')
    ax1.set_xlim([0, 1])
    ax2.set_xlim([0, 1])
    plt.show()
