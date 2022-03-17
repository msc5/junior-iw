
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gen_lin(sl: int = 100):
    # Generates a line with displacement (0, 0.5)
    # and random slope (-0.5, 0.5]
    a = 0.5
    m = np.random.rand() - 0.5
    t = np.linspace(0, 1, sl)
    d = m * t + a
    return d


def gen_lins(si: int = 10, sl: int = 50, fl: int = 50):
    # Generates (si) lines with length sl
    data = np.stack([gen_lin(sl + fl) for _ in range(si)], axis=1)
    data = np.expand_dims(data, 0)
    return data[:, :sl], data[:, sl:]


def gen_sin(sl: int = 100):
    # Generates a sin wave with a random period (0, 2]
    # and a random phase (0, pi]. (All have length sl)
    a = np.random.rand() * 4 * np.pi
    b = np.random.rand() * np.pi
    t = np.linspace(0, 1, sl)
    d = (np.sin(a * t + b) + 1) / 2
    return d


def gen_sins(si: int = 10, sl: int = 50, fl: int = 50):
    # Generates (si) sin waves with random period and phase
    # with length sl
    data = np.stack([gen_sin(sl + fl) for _ in range(si)], axis=1)
    data = np.expand_dims(data, 0)
    return data[:, :sl], data[:, sl:]


def plot_seq(x_train, y_train, output):
    print('Train: ', x_train.shape)
    print('Test: ', y_train.shape)
    print('Output: ', output.shape)
    assert y_train.shape == output.shape
    _, sl, si = x_train.shape
    fl = y_train.shape[1]
    t = np.linspace(0, 1, sl * 2)
    st, ft = t[:sl], t[sl:sl + fl]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    lines = []
    for n in range(si):
        ax1.plot(st, x_train[0, :, n], color='blue')
        ax1.plot(ft, y_train[0, :, n], color='blue')
        ax2.plot(st, x_train[0, :, n], color='blue')
        ax2.plot(ft, y_train[0, :, n], color='limegreen')
        lines += [ax2.plot(ft, output[0, :, n], color='magenta')][0]
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    return fig, lines


def plot_seq_img(x_train, y_train, output):
    pass


if __name__ == '__main__':

    bs = 1
    si = 10
    sl = 100
    fl = sl

    print(f'{"Input Size":15}{si:5}')
    print(f'{"Sequence Length":15}{sl:5}')
    print(f'{"Future Length":15}{fl:5}')
    print(f'{"-->":15}({1:3}, {sl:3}, {si:3})')
    print(f'{"-->":15}({1:3}, {fl:3}, {si:3})')

    # x = gen_sin(sl)
    # a, b = gen_sins(si, sl, fl)
    # print('gen_sins()')
    # print(x.shape)
    # print(a.shape, b.shape)
    # plot_seq(a, b, b)
    # plt.show()

    # x = gen_lin(sl)
    # a, b = gen_lins(si, sl, fl)
    # print('gen_lins()')
    # print(x.shape)
    # print(a.shape, b.shape)
    # plot_seq(a, b, b)
    # plt.show()

    a, b = gen_sins(si**2, sl, fl)
    print('gen_sins()')
    print(a.shape, b.shape)
    a = a.reshape((bs, sl, si, si))
    b = b.reshape((bs, fl, si, si))

    fig, ax = plt.subplots()
    frame = ax.imshow(a[0, 0])

    def video(i):
        frame.set_data(a[0, i])
        return frame

    ani = animation.FuncAnimation(
        fig,
        video,
        interval=1,
    )
    plt.show()
