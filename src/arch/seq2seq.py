
import torch
import torch.nn as nn


class Seq2Seq (nn.Module):

    def __init__(self, enc_cell: nn.Module, dec_cell: nn.Module):
        super(Seq2Seq, self).__init__()
        self.enc_cell, self.dec_cell = enc_cell, dec_cell

    def forward(self, x, prediction_len: int = None):

        # x : (batch_size, seq_len, img_chan, img_h, img_w)
        batch_size, seq_len, img_chan, img_h, img_w = x.shape
        fut_len = prediction_len or seq_len

        self.enc_cell.init_params(x)
        self.dec_cell.init_params(x)

        def pass_through(
                cell: nn.ModuleList,  # Encoder or Decoder for each depth
                x: torch.Tensor,       # Input data
        ):
            h, c = cell.enc_h, cell.enc_c
            h[0], c[0] = cell[0](x, (h[0], c[0]))
            for e in range(1, len(h)):
                h[e], c[e] = cell[e](h[e - 1], (h[e], c[e]))
            return h[-1]

        output = torch.zeros(
            (fut_len, batch_size, self.hid_chan, img_h, img_w),
            device=x.device)

        for t in range(seq_len):
            state = pass_through(self.enc_cell, x[:, t])

        for t in range(fut_len):
            state = pass_through(self.dec_cell, state)
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


if __name__ == "__main__":
    pass
