
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

    def forward(self, x, hidden: tuple):
        bs, si = x.shape        # (batch size, input size)
        h, c = hidden
        c = self.f(x, h) * c + self.i(x, h) * self.g(x, h)
        h = self.o(x, h) * torch.tanh(c)
        return nn.Parameter(h), nn.Parameter(c)


class Seq2SeqLSTM (nn.Module):

    def __init__(
        self,
        bs: int = 1,
        sizes: (int, int) = (10, 20),
        depths: (int, int) = (1, 1),
    ):
        super(Seq2SeqLSTM, self).__init__()
        self.bs = bs
        self.si, self.sh = sizes
        self.de, self.dd = depths
        self.init_layers()
        self.init_params()

    def init_layers(self) -> None:
        def init_layer(si: int, d: int):
            return nn.ModuleList(
                [LSTMCell(si, self.sh)] +
                [LSTMCell(self.sh, self.sh) for _ in range(d)]
                #  [nn.LSTMCell(si, self.sh)] +
                #  [nn.LSTMCell(self.sh, self.sh) for _ in range(d)]
            )
        self.enc = init_layer(self.si, self.de)
        self.dec = init_layer(self.sh, self.dd)
        self.fin = nn.Linear(self.sh, self.si)

    def init_params(self) -> None:
        def init_param(n: int):
            params = []
            for i in range(n):
                #  param = torch.zeros(self.bs, self.sh)
                param = torch.rand(self.bs, self.sh)
                param = nn.Parameter(param)
                params += [param]
            return nn.ParameterList(params)
        self.enc_h = init_param(self.de + 1)
        self.enc_c = init_param(self.de + 1)
        self.dec_h = init_param(self.dd + 1)
        self.dec_c = init_param(self.dd + 1)

    def reset_params(self) -> None:
        def reset_param(param):
            for layer in param:
                layer = layer * 0
        reset_param(self.enc_h)
        reset_param(self.enc_c)
        reset_param(self.dec_h)
        reset_param(self.dec_h)

    def forward(self, x, fl: int = None):

        bs, sl, si = x.shape     # (batch size, time step, input size)
        fl = sl if fl is None else fl

        output = []
        self.reset_params()

        def pass_through(
                layers: nn.ModuleList,
                h: nn.ParameterList,
                c: nn.ParameterList,
                x: torch.Tensor
        ):
            assert len(h) == len(c)
            assert h[0].shape == c[0].shape
            h[0], c[0] = layers[0](x, (h[0], c[0]))
            for e in range(1, len(h)):
                h[e], c[e] = layers[e](h[e - 1], (h[e], c[e]))
            return h[len(h) - 1]

        for t in range(sl):
            state = pass_through(self.enc, self.enc_h, self.enc_c, x[:, t, :])

        for t in range(fl):
            #  start = [sum([torch.linalg.norm(H) for H in self.dec_h]),
            #  sum([torch.linalg.norm(C) for C in self.dec_c])]
            state = pass_through(self.dec, self.enc_h, self.enc_c, state)
            output += [state]
            #  stop = [sum([torch.linalg.norm(H) for H in self.dec_h]),
            #  sum([torch.linalg.norm(C) for C in self.dec_c])]
            #  dx = [b.item() - a.item() for a, b in zip(start, stop)]
            #  print(dx)

        output = torch.stack(output)            # --> (sl, bs, sh)
        output = output.permute(1, 0, 2)        # --> (bs, sl, sh)
        output = self.fin(output)               # --> (bs, sl, si
        output = torch.nn.Sigmoid()(output)     # --> range: (0, 1)

        return output


if __name__ == '__main__':

    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bs = 16
    sl = 30

    # sizes
    si = 10
    sh = 20

    # depths
    de = 3
    dd = 3

    model = Seq2SeqLSTM(bs, (si, sh), (de, dd)).to(device)
    summary(model, input_size=(bs, sl, si))
