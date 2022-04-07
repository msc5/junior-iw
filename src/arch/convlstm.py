
import torch
import torch.nn as nn

from collections.abc import Callable


class ConvLSTMGate (nn.Module):

    def __init__(
            self,
            inp_chan: int,
            out_chan: int,
            cell_state: bool = True,
    ):
        super(ConvLSTMGate, self).__init__()
        self.inp_chan = inp_chan
        self.out_chan = out_chan
        self.layer = nn.Conv2d(
            in_channels=(self.inp_chan + self.out_chan),
            out_channels=(4 * self.out_chan),
            kernel_size=(3, 3), padding='same', bias=True
        )
        self.acts = nn.ModuleList([
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Sigmoid()
        ])

    def forward(self, x, hidden: tuple):
        # x : (batch_size, img_chan, img_h, img_w)
        # hidden : (batch_size, out_chan, img_h, img_w)
        h, c = hidden
        combined = self.layer(torch.cat([x, h], dim=1))
        gates = torch.split(combined, self.out_chan, dim=1)
        i, f, g, o = [self.acts[j](G) for j, G in enumerate(gates)]
        return i, f, g, o


class ConvLSTMCell (nn.Module):

    def __init__(self, inp_chan: int, out_chan: int):
        super(ConvLSTMCell, self).__init__()
        self.gates = ConvLSTMGate(inp_chan, out_chan)

    def forward(self, x, hidden: tuple):
        # x : (batch_size, inp_chan, img_h, img_w)
        # hidden : (batch_size, out_chan, img_h, img_w)

        # Generate gate outputs
        h, c = hidden
        i, f, g, o = self.gates(x, hidden)

        # Compute new c and h
        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTMSeq2Seq (nn.Module):

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        model_depth: int = 1,
    ):
        super(ConvLSTMSeq2Seq, self).__init__()
        self.inp_chan = input_channels
        self.hid_chan = hidden_channels
        self.model_dep = model_depth
        self.init_layers()

    def init_layers(self) -> None:
        def init_layer(inp_chan: int, depth: int):
            return nn.ModuleList(
                [ConvLSTMCell(inp_chan, self.hid_chan)] +
                [ConvLSTMCell(self.hid_chan, self.hid_chan)
                 for _ in range(depth)]
            )
        self.enc = init_layer(self.inp_chan, self.model_dep)
        self.dec = init_layer(self.hid_chan, self.model_dep)
        self.fin = nn.Conv3d(
            self.hid_chan, self.inp_chan,
            (1, 3, 3), padding=(0, 1, 1)
        )

    def init_params(self, x: torch.Tensor) -> None:
        def init_param(n: int):
            batch_size, _, _, img_h, img_w = x.shape
            param_shape = (batch_size, self.hid_chan, img_h, img_w)
            params = [torch.zeros(*param_shape, device=x.device)
                      for _ in range(n + 1)]
            return params
        self.enc_h = init_param(self.model_dep)
        self.enc_c = init_param(self.model_dep)
        self.dec_h = init_param(self.model_dep)
        self.dec_c = init_param(self.model_dep)

    def forward(self, x, prediction_len: int = None):

        # x : (batch_size, seq_len, img_chan, img_h, img_w)
        batch_size, seq_len, img_chan, img_h, img_w = x.shape
        fut_len = prediction_len or seq_len

        self.init_params(x)

        def pass_through(
                layers: nn.ModuleList,  # Encoder or Decoder for each depth
                x: torch.Tensor,       # Input data
                h: list,               # hidden layer for each depth
                c: list,               # cell state for each depth
        ):
            h[0], c[0] = layers[0](x, (h[0], c[0]))
            for e in range(1, len(h)):
                h[e], c[e] = layers[e](h[e - 1], (h[e], c[e]))
            return h[-1]

        output = torch.zeros(
            (fut_len, batch_size, self.hid_chan, img_h, img_w),
            device=x.device)

        for t in range(seq_len):
            state = pass_through(self.enc, x[:, t], self.enc_h, self.enc_c)

        for t in range(fut_len):
            state = pass_through(self.dec, state, self.dec_h, self.dec_c)
            output[t] = state

        # (fut_len, batch_size, hid_chan, img_h, img_w)
        output = output.permute(1, 2, 0, 3, 4)
        # (batch_size, hid_chan, fut_len, img_h, img_w)
        output = self.fin(output)
        # (batch_size, img_chan, fut_len, img_h, img_w)
        output = output.permute(0, 2, 1, 3, 4)
        # (batch_size, fut_len, img_chan, img_h, img_w)
        output = torch.sigmoid(output)
        # --> range: (0, 1)

        return output


if __name__ == '__main__':

    from rich import print
    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_shape = (4, 10, 1, 64, 64)
    model = ConvLSTMSeq2Seq(1, 64, 3).to(device)

    # summary(model, input_size=x_shape))

    x = torch.rand(*x_shape).to(device)
    output = model(x)
