
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
        self.W_x = nn.Linear(inp_size, out_size, bias=False)
        self.W_h = nn.Linear(out_size, out_size, bias=False)
        self.W_c = nn.Parameter(torch.rand(out_size))
        self.bias = nn.Parameter(torch.rand(out_size))
        self.activation = activation()

    def forward(self, x, hidden: tuple):
        h, c = hidden
        return self.activation(
            self.W_x(x) + self.W_h(h) +
            self.W_c * c + self.bias)


class LSTMCell (nn.Module):

    def __init__(self, inp_size: int, out_size: int):
        super(LSTMCell, self).__init__()
        self.f = LSTMGate(inp_size, out_size, nn.Sigmoid)
        self.i = LSTMGate(inp_size, out_size, nn.Sigmoid)
        self.g = LSTMGate(inp_size, out_size, nn.Tanh, cell_state=False)
        self.o = LSTMGate(inp_size, out_size, nn.Sigmoid)

    def forward(self, x, hidden: tuple):
        h, c = hidden
        c = self.f(x, hidden) * c + self.i(x, hidden) * self.g(x, hidden)
        h = self.o(x, hidden) * torch.tanh(c)
        return nn.Parameter(h), nn.Parameter(c)


class Seq2SeqLSTM (nn.Module):

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        model_depth: int,
    ):
        super(Seq2SeqLSTM, self).__init__()

        # Define global Variables
        global hid_size
        hid_size = hidden_size
        global dat_size
        dat_size = data_size
        global model_dep
        model_dep = model_depth

        self.init_layers()
        self.init_params()

    def init_layers(self) -> None:
        def init_layer(inp_size: int, d: int):
            return nn.ModuleList(
                [LSTMCell(inp_size, hid_size)] +
                [LSTMCell(hid_size, hid_size) for _ in range(d)]
            )
        self.enc = init_layer(dat_size, model_dep)
        self.dec = init_layer(hid_size, model_dep)
        self.fin = nn.Linear(hid_size, dat_size)

    def init_params(self) -> None:
        def init_param(n: int):
            params = []
            for i in range(n):
                param = torch.rand(1, hid_size)
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
        self.reset_params()

        def pass_through(
                layers: nn.ModuleList,
                h: nn.ParameterList,
                c: nn.ParameterList,
                x: torch.Tensor
        ):
            h[0], c[0] = layers[0](x, (h[0], c[0]))
            for e in range(1, len(h)):
                h[e], c[e] = layers[e](h[e - 1], (h[e], c[e]))
            return h[len(h) - 1]

        for t in range(seq_len):
            state = pass_through(self.enc, self.enc_h, self.enc_c, x[:, t])

        for t in range(pred_len):
            state = pass_through(self.dec, self.enc_h, self.enc_c, state)
            output += [state.squeeze(0)]

        output = torch.stack(output)        # --> (pred_len, hid_size)
        output = self.fin(output)           # --> (pred_len, dat_size)
        output = torch.sigmoid(output)      # --> range: (0, 1)

        return output.unsqueeze(0)


if __name__ == '__main__':

    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = 30

    dat_size = 10
    hid_size = 20

    model_dep = 3

    model = Seq2SeqLSTM(dat_size, hid_size, model_dep).to(device)
    summary(model, input_size=(1, seq_len, dat_size))
