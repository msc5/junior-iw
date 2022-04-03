
import torch
import torch.nn as nn

from collections.abc import Callable


class ConvLSTMGate (nn.Module):

    def __init__(
            self,
            inp_chan: int,
            out_chan: int,
            activation: Callable = nn.Sigmoid,
            cell_state: bool = True,
    ):
        super(ConvLSTMGate, self).__init__()
        self.W_x = nn.Conv2d(inp_chan, out_chan, 3, padding='same', bias=False)
        self.W_h = nn.Conv2d(out_chan, out_chan, 3, padding='same', bias=False)
        self.W_c = nn.Parameter(torch.zeros(1, out_chan, img_h, img_w))
        self.bias = nn.Parameter(torch.zeros(1, out_chan, img_h, img_w))
        self.activation = activation()

    def forward(self, x, hidden: tuple):
        # x : (bat_size, seq_len, img_chan, img_h, img_w)
        # h, c : (1, img_chan, img_h, img_w)
        h, c = hidden
        # print(x.shape, h.shape, c.shape)
        return self.activation(
            self.W_x(x) + self.W_h(h) +
            self.W_c * c + self.bias)


class ConvLSTMCell (nn.Module):

    def __init__(self, inp_chan: int, out_chan: int):
        super(ConvLSTMCell, self).__init__()
        self.i = ConvLSTMGate(inp_chan, out_chan, nn.Sigmoid)
        self.f = ConvLSTMGate(inp_chan, out_chan, nn.Sigmoid)
        self.g = ConvLSTMGate(inp_chan, out_chan, nn.Tanh, cell_state=False)
        self.o = ConvLSTMGate(inp_chan, out_chan, nn.Sigmoid)

    def forward(self, x, hidden: tuple):
        h, c = hidden
        c = self.f(x, hidden) * c + self.i(x, hidden) * self.g(x, hidden)
        h = self.o(x, hidden) * torch.tanh(c)
        return nn.Parameter(h), nn.Parameter(c)


class ConvLSTMSeq2Seq (nn.Module):

    def __init__(
        self,
        hidden_channels: int,
        img_shape: (int, int, int),         # (img_chan, img_h, img_w)
        model_depth: int = 1,
    ):
        super(ConvLSTMSeq2Seq, self).__init__()
        global hid_chan
        hid_chan = hidden_channels
        global img_chan, img_h, img_w
        img_chan, img_h, img_w = img_shape
        global model_dep
        model_dep = model_depth
        self.init_layers()

    def init_layers(self) -> None:
        def init_layer(inp_chan: int, depth: int):
            return nn.ModuleList(
                [ConvLSTMCell(inp_chan, hid_chan)] +
                [ConvLSTMCell(hid_chan, hid_chan) for _ in range(depth)]
            )
        self.enc = init_layer(img_chan, model_dep)
        self.dec = init_layer(hid_chan, model_dep)
        self.fin = nn.Conv3d(hid_chan, 1, (1, 3, 3), padding='same')
        self.norm = nn.BatchNorm2d(1)
        self.sigmoid = torch.nn.Sigmoid()

    def init_params(self, batch_size: int) -> None:
        def init_param(n: int):
            params = []
            for i in range(n):
                param = torch.zeros(batch_size, hid_chan, img_h, img_w)
                params += [nn.Parameter(param)]
            return nn.ParameterList(params)
        self.enc_h = init_param(model_dep + 1)
        self.enc_c = init_param(model_dep + 1)
        self.dec_h = init_param(model_dep + 1)
        self.dec_c = init_param(model_dep + 1)

    def reset_params(self) -> None:
        def reset_param(param):
            for layer in param:
                layer = layer * 0
        reset_param(self.enc_h)
        reset_param(self.enc_c)
        reset_param(self.dec_h)
        reset_param(self.dec_c)

    def forward(self, x, prediction_len: int = None):

        # x --> (bat_size, seq_len, img_chan, img_h, img_w)
        batch_size, seq_len = x.shape[0], x.shape[1]
        fut_len = prediction_len or seq_len

        output = []
        self.init_params(batch_size)

        def pass_through(
                layers: nn.ModuleList,      # Encoder or Decoder for each depth
                h: nn.ParameterList,        # hidden layer for each depth
                c: nn.ParameterList,        # cell state for each depth
                x: torch.Tensor             # Input data
        ):
            h[0], c[0] = layers[0](x, (h[0], c[0]))
            for e in range(1, len(h)):
                h[e], c[e] = layers[e](h[e - 1], (h[e], c[e]))
            return h[len(h) - 1]

        for t in range(seq_len):
            state = pass_through(self.enc, self.enc_h, self.enc_c, x[:, t])

        for t in range(fut_len):
            state = pass_through(self.dec, self.dec_h, self.dec_c, state)
            output += [state]

        output = torch.stack(output)
        # (fut_len, batch_size, hid_chan, img_h, img_w)
        output = output.permute(1, 2, 0, 3, 4)
        # (batch_size, hid_chan, fut_len, img_h, img_w)
        output = self.fin(output)
        # (batch_size, img_chan, fut_len, img_h, img_w)
        output = output.permute(0, 2, 1, 3, 4)
        # (batch_size, fut_len, img_chan, img_h, img_w)
        output = self.sigmoid(output)
        # --> range: (0, 1)

        return output


if __name__ == '__main__':

    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    seq_len = 10
    x_shape = (1, 64, 64)
    # x_shape = (1, 64, 64)

    model_dep = 3
    hid_chan = 15

    model = ConvLSTMSeq2Seq(hid_chan, x_shape, model_dep).to(device)

    # summary(model, input_size=(1, seq_len, *x_shape))

    x = torch.rand(batch_size, seq_len, *x_shape).to(device)
    output = model(x)
