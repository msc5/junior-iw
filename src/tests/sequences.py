
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


def gen_sin(seq_len: int = 100):
    # Generates a sin wave with a random period (0, 2]
    # and a random phase (0, pi]. (All have length seq_len)
    a = np.random.rand() * 4 * np.pi
    b = np.random.rand() * np.pi
    t = np.linspace(0, 1, seq_len)
    d = (np.sin(a * t + b) + 1) / 2
    return d


def gen_sins(batch_size: int, inp_size: int, seq_len: int, fut_len: int):
    # Generates (inp_size) sin waves with random period and phase
    # with length seq_len
    data = np.stack([np.stack([gen_sin(seq_len + fut_len)
                     for _ in range(inp_size)])
                     for _ in range(batch_size)])
    data = data.transpose(0, 2, 1)
    return data[:, :seq_len, :], data[:, seq_len:, :]


def plot_seq(x_train, y_train, output):
    print(f'{"Train:":>10} : {str(x_train.shape):<10}')
    print(f'{"Test:":>10} : {str(y_train.shape):<10}')
    print(f'{"Output:":>10} : {str(output.shape):<10}')
    assert y_train.shape == output.shape
    _, seq_len, inp_size = x_train.shape
    _, fut_len, _ = y_train.shape
    t = np.linspace(0, 1, seq_len + fut_len)
    st, ft = t[:seq_len], t[seq_len:(seq_len + fut_len)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    lines = []
    for n in range(inp_size):
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


if __name__ == '__main__':

    batch_size = 10
    inp_size = 5
    seq_len = 20
    fut_len = seq_len

    print(f'{"Input Size":15}{inp_size:5}')
    print(f'{"Sequence Length":15}{seq_len:5}')
    print(f'{"Future Length":15}{fut_len:5}')
    print(f'{"-->":15}({1:3}, {seq_len:3}, {inp_size:3})')
    print(f'{"-->":15}({1:3}, {fut_len:3}, {inp_size:3})')

    # -------------------------------------------------------------------------
    # Test gen_sins() and plot_seq()
    # -------------------------------------------------------------------------

    x = gen_sin(seq_len)
    a, b = gen_sins(batch_size, inp_size, seq_len, fut_len)
    print('gen_sins()')
    print(x.shape)
    print(a.shape, b.shape)
    plot_seq(a, b, b)
    plt.show()

    # -------------------------------------------------------------------------
    # Test gen_lins() and plot_seq()
    # -------------------------------------------------------------------------

    # x = gen_lin(seq_len)
    # a, b = gen_lins(inp_size, seq_len, fut_len)
    # print('gen_lins()')
    # print(x.shape)
    # print(a.shape, b.shape)
    # plot_seq(a, b, b)
    # plt.show()

    # -------------------------------------------------------------------------
    # Test prediction_video()
    # -------------------------------------------------------------------------

    # a, b = gen_sins(si**2, sl, fl)
    # print('gen_sins()')
    # print(a.shape, b.shape)
    # a = a.reshape((bs, sl, si, si))
    # b = b.reshape((bs, fl, si, si))
    # prediction_video(a, b, (b + 7) / 10)
