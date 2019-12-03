import torch
import torch.nn as nn

from torch.autograd import Variable


class TrackNet(nn.Module):
    def __init__(self):
        super(TrackNet, self).__init__()

        # TODO: some changes if GPU is available?

        self.conv1d = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(32)
        self.conv1d_relu = nn.ReLU()
        self.gru1 = nn.GRU(input_size=32, hidden_size=32)
        self.gru2 = nn.GRU(input_size=32, hidden_size=16)
        self.linear1 = nn.Linear(16, 2)
        self.linear2 = nn.Linear(16, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.conv1d_relu(x)
        x = x.permute(2, 0, 1)
        x, hn = self.gru1(x)
        x, hn = self.gru2(x)
        x = x[x.size()[0] - 1]
        xy_coords = self.linear1(x)
        x = self.linear2(x)
        r1_r2 = self.softplus(x)
        result = torch.cat((xy_coords, r1_r2), dim=1)
        return result