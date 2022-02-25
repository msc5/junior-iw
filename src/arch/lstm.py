
import torch
import torch.nn as nn

from collections.abc import Callable


class LSTMGate (nn.Module):

    def __init__(
            self,
            si: int = 10,
            sh: int = 20,
            op: Callable = nn.Linear,
            act: Callable = nn.Sigmoid,
    ):
        super(LSTMGate, self).__init__()
        self.U = op(si, sh, bias=True)
        self.V = op(sh, sh, bias=False)
        self.act = act()

    def forward(self, x, h):
        return self.act(self.U(x) + self.V(h))


class LSTMCell (nn.Module):

    def __init__(
            self,
            si: int = 10,
            sh: int = 20
    ):
        super(LSTMCell, self).__init__()
        self.si = si
        self.sh = sh
        self.f = LSTMGate(si, sh, nn.Linear, nn.Sigmoid)
        self.i = LSTMGate(si, sh, nn.Linear, nn.Sigmoid)
        self.g = LSTMGate(si, sh, nn.Linear, nn.Tanh)
        self.o = LSTMGate(si, sh, nn.Linear, nn.Sigmoid)

    def forward(self, x, h, c):
        bs, si = x.shape        # (batch size, input size)
        C = self.f(x, h) * c + self.i(x, h) * self.g(x, h)
        H = self.o(x, h) * torch.tanh(C)
        return nn.Parameter(H), nn.Parameter(C)


class Seq2SeqLSTM (nn.Module):

    def __init__(
        self,
        bs: int = 1,
        sl: int = 10,
        fl: int = 10,
        ne: int = 4,
        de: int = 1,
        nd: int = 4,
        dd: int = 1,
        si: int = 10,
        sh: int = 20,
    ):
        super(Seq2SeqLSTM, self).__init__()
        self.bs = bs
        self.sl = sl
        self.fl = fl
        self.ne = ne        # Number of Cells in Encoder
        self.de = de        # Depth of the Encoder
        self.nd = nd
        self.dd = dd
        self.si = si
        self.sh = sh

        def init_layers(n: int, d: int):
            return [nn.ModuleList(
                    [LSTMCell(self.si, self.sh)] +
                    [LSTMCell(self.sh, self.sh) for _ in range(n)]
                ) for _ in range(d)]

        self.enc = init_layers(self.ne, self.de)
        self.dec = init_layers(self.nd, self.dd)

        self.fin = nn.Linear(self.sh, self.si)

        def init_weights(n: int):
            weights = []
            for i in range(n):
                weight = torch.zeros((self.bs, self.sh))
                weight = nn.Parameter(weight)
                weights += [weight]
            return nn.ParameterList(weights)

        self.enc_h = init_weights(self.de + 1)
        self.enc_c = init_weights(self.de + 1)
        self.dec_h = init_weights(self.dd + 1)
        self.dec_c = init_weights(self.dd + 1)

    def forward(self, x):

        bs, T, si = x.shape     # (batch size, time step, input size)

        output = []

        for t in range(T):
            print(x[:, t, :].shape)
            print(self.enc_h[0].shape)
            print(self.enc_c[0].shape)
            for n in self.enc:
                self.enc_h[0], self.enc_c[0] = n[0](
                    x[:, t, :],
                    self.enc_h[0],
                    self.enc_c[0]
                )
                for e in range(1, self.de):
                    self.enc_h[e], self.enc_c[e] = n[e](
                        self.enc_h[e - 1],
                        self.enc_h[e],
                        self.enc_c[e]
                    )

        state = self.enc_h[-1]

        for t in range(self.fl):
            for n in self.dec:
                self.dec_h[0], self.dec_c[0] = n[0](
                    state,
                    self.dec_h[0],
                    self.dec_c[0]
                )
                for e in range(1, self.dd):
                    self.dec_h[e], self.dec_c[e] = n[e](
                        self.dec_h[e - 1],
                        self.dec_h[e],
                        self.dec_c[e]
                    )
                state = self.dec_h[-1]
                output += [state]

        output = torch.stack(output)            # --> (T, bs, sh)
        output = output.permute(1, 0, 2)        # --> (bs, T, sh)
        output = self.fin(output)               # --> (bs, T, si)
        output = torch.nn.Sigmoid()(output)

        return output


if __name__ == '__main__':

    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (bs, T, si)
    si = 10
    sh = 20
    bs = 16
    T = 30

    model = Seq2SeqLSTM(bs, T, T, 1, 1, si, sh).to(device)
    summary(model, input_size=(bs, T * 2, si))
