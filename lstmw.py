from torch import nn
import torch


class LstmwT(nn.Module):
    """
        A LSTM Module including a batch normalize layer, a linear input layer, a lstm, and a linear output layer.
    """

    def __init__(self, output_dim, pcseq):
        super().__init__()
        self.bn = nn.BatchNorm1d(30)
        self.fc_in = nn.Linear(30, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256,
                            num_layers=6, batch_first=True)
        self.pcseq = pcseq
        self.fc_out = nn.Linear(6 * 256 * 2, output_dim)

    def forward(self, x):
        x = x[:, -self.pcseq:, :]
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc_in(x)
        output, (h_n, c_n) = self.lstm(x)
        hc = torch.cat((h_n, c_n), dim=0)
        hc = hc.permute(1, 0, 2)
        hc = hc.reshape(hc.shape[0], -1)
        out = self.fc_out(hc)
        return out
