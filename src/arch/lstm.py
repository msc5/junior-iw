
import torch
import torch.nn as nn

from collections.abc import Callable


class LSTMGate (nn.Module):

    def __init__(
            self,
            inp_size: int,
            out_size: int,
            activation: Callable = nn.Sigmoid,
            cell_state: bool = True,
    ):
        super(LSTMGate, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.layer = nn.Linear(
            in_features=(self.inp_size + self.out_size),
            out_features=(4 * self.out_size), bias=True)
        self.acts = nn.ModuleList([
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Sigmoid()
        ])

    def forward(self, x, hidden: tuple):
        # x : (batch_size, img_size, dat_size)
        # hidden : (batch_size, out_size, dat_size)
        h, c = hidden
        combined = self.layer(torch.cat([x, h], dim=1))
        gates = torch.split(combined, self.out_size, dim=1)
        i, f, g, o = [self.acts[j](G) for j, G in enumerate(gates)]
        return i, f, g, o


class LSTMCell (nn.Module):

    def __init__(self, inp_size: int, out_size: int):
        super(LSTMCell, self).__init__()
        self.gates = LSTMGate(inp_size, out_size)

    def forward(self, x, hidden: tuple):
        # x : (batch_size, seq_len, dat_size)
        # hidden : (batch_size, hid_size, dat_size)
        h, c = hidden

        # Generate gate outputs
        h, c = hidden
        i, f, g, o = self.gates(x, hidden)

        # Compute new c and h
        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class LSTMSeq2Seq (nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        model_depth: int = 1,
    ):
        super(LSTMSeq2Seq, self).__init__()
        self.inp_size = input_size
        self.hid_size = hidden_size
        self.model_dep = model_depth
        self.init_layers()

    def init_layers(self) -> None:
        def init_layer(inp_size: int, depth: int):
            return nn.ModuleList(
                [LSTMCell(inp_size, self.hid_size)] +
                [LSTMCell(self.hid_size, self.hid_size)
                    for _ in range(depth)]
            )
        self.enc = init_layer(self.inp_size, self.model_dep)
        self.dec = init_layer(self.hid_size, self.model_dep)
        self.fin = nn.Linear(self.hid_size, self.inp_size)

    def init_params(self, x: torch.Tensor) -> None:
        def init_param(n: int):
            batch_size, _, dat_size = x.shape
            param_shape = (batch_size, self.hid_size)
            params = [torch.zeros(*param_shape, device=x.device)
                      for _ in range(n + 1)]
            return params
        self.enc_h = init_param(self.model_dep)
        self.enc_c = init_param(self.model_dep)
        self.dec_h = init_param(self.model_dep)
        self.dec_c = init_param(self.model_dep)

    def forward(self, x, prediction_len: int = None):

        # x : (batch_size, seq_len, dat_size)
        batch_size, seq_len, dat_size = x.shape
        fut_len = prediction_len or seq_len

        self.init_params(x)

        def pass_through(
                layers: nn.ModuleList,
                x: torch.Tensor,
                h: list[torch.Tensor],
                c: list[torch.Tensor],
        ):
            h[0], c[0] = layers[0](x, (h[0], c[0]))
            for e in range(1, len(h)):
                h[e], c[e] = layers[e](h[e - 1], (h[e], c[e]))
            return h[len(h) - 1]

        output = torch.zeros(
            (fut_len, batch_size, self.hid_size), device=x.device)

        for t in range(seq_len):
            state = pass_through(self.enc, x[:, t], self.enc_h, self.enc_c)

        for t in range(fut_len):
            state = pass_through(self.dec, state, self.dec_h, self.dec_c)
            output[t] = state

        # (fut_len, batch_size, hid_size)
        output = self.fin(output)
        # (fut_len, batch_size, dat_size)
        output = output.permute(1, 0, 2)
        # (batch_size, fut_len, dat_size)
        output = torch.sigmoid(output)
        # --> range: (0, 1)

        return output


if __name__ == '__main__':

    from rich import print
    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    seq_len = 20
    dat_size = 1
    x_shape = (batch_size, seq_len, dat_size)

    model = LSTMSeq2Seq(dat_size, 64, 1).to(device)

    summary(model, input_size=x_shape)

    x = torch.rand(*x_shape).to(device)
    output = model(x)
