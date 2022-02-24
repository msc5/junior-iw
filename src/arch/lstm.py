
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


class LSTM (nn.Module):

    def __init__(
            self,
            si: int = 10,
            sh: int = 20
    ):
        super(LSTM, self).__init__()
        self.si = si
        self.sh = sh
        self.f = LSTMGate(si, sh, nn.Linear, nn.Sigmoid)
        self.i = LSTMGate(si, sh, nn.Linear, nn.Sigmoid)
        self.c = LSTMGate(si, sh, nn.Linear, nn.Tanh)
        self.o = LSTMGate(si, sh, nn.Linear, nn.Sigmoid)

    def forward(self, X):
        # (batch size, sequence length, input size)
        bs, T, _ = X.size()
        sequence = []
        h, c = (
            torch.zeros(bs, self.sh).to(X.device),
            torch.zeros(bs, self.sh).to(X.device),
        )
        for t in range(T):
            x = X[:, t, :]
            c = self.f(x, h) * c + self.i(x, h) * self.c(x, h)
            h = self.o(x, h) * torch.tanh(c)
            sequence.append(h.unsqueeze(0))
        sequence = torch.cat(sequence, dim=0)
        return sequence, (h, c)


if __name__ == '__main__':

    from torchinfo import summary

    si = 10
    sh = 20
    bs = 16
    T = 100

    model = LSTM(si, sh)
    summary(model, input_size=(bs, T, si))
