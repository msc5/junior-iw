
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
        self.W_c = nn.Parameter(torch.rand(out_chan, img_h, img_w))
        self.bias = nn.Parameter(torch.rand(out_chan, img_h, img_w))
        self.activation = activation()

    def forward(self, x, hidden: tuple):
        h, c = hidden
        return self.activation(
            self.W_x(x) + self.W_h(h) +
            self.W_c * c + self.bias)


class ConvLSTMCell (nn.Module):

    def __init__(self, inp_chan: int, out_chan: int):
        super(ConvLSTMCell, self).__init__()
        self.f = ConvLSTMGate(inp_chan, out_chan, nn.Sigmoid)
        self.i = ConvLSTMGate(inp_chan, out_chan, nn.Sigmoid)
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
        model_depth: int = 1
    ):
        super(ConvLSTMSeq2Seq, self).__init__()

        # Define global Variables
        global hid_chan
        hid_chan = hidden_channels
        global img_chan, img_h, img_w
        img_chan, img_h, img_w = img_shape
        global model_dep
        model_dep = model_depth

        self.init_layers()
        self.init_params()

    def init_layers(self) -> None:
        def init_layer(inp_chan: int, depth: int):
            return nn.ModuleList(
                [ConvLSTMCell(inp_chan, hid_chan)] +
                [ConvLSTMCell(hid_chan, hid_chan) for _ in range(depth)]
            )
        self.enc = init_layer(img_chan, model_dep)
        self.dec = init_layer(hid_chan, model_dep)
        self.fin = nn.Conv2d(hid_chan, img_chan, 3, padding='same')

    def init_params(self) -> None:
        def init_param(n: int):
            params = []
            for i in range(n):
                param = torch.rand(1, hid_chan, img_h, img_w)
                params += [nn.Parameter(param)]
            return nn.ParameterList(params)
        self.enc_h = init_param(model_dep + 1)
        self.enc_c = init_param(model_dep + 1)

    def reset_params(self) -> None:
        def reset_param(param):
            for layer in param:
                layer = layer * 0
        reset_param(self.enc_h)
        reset_param(self.enc_c)

    def forward(self, x, pred_len: int = None):

        # x -> (seq_len, img_chan, img_h, img_w)
        seq_len = len(x)
        pred_len = seq_len if pred_len is None else pred_len

        output = []
        self.reset_params()  # ?

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

        for t in range(pred_len):
            state = pass_through(self.dec, self.enc_h, self.enc_c, state)
            print(state.detach().min(), state.detach().max())
            output += [state.squeeze(0)]

        output = torch.stack(output)  # --> (pred_len, hid_chan, img_h, img_w)
        output = self.fin(output)     # --> (pred_len, img_chan, img_h, img_w)
        output = torch.sigmoid(output)      # --> range: (0, 1)
        print(output.detach().min(), output.detach().max())

        return output.unsqueeze(0)


if __name__ == '__main__':

    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = 30
    x_shape = (3, 50, 100)

    model_dep = 3
    hid_chan = 20

    model = ConvLSTMSeq2Seq(hid_chan, x_shape, model_dep).to(device)
    summary(model, input_size=(1, seq_len, *x_shape))
