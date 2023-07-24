from . import ResNetConv
import torch
from torch import nn

class ResNetLSTM(nn.Module):
    def __init__(self, winsize: int):
        super().__init__()
        self.winsize = winsize

        self.resnet = ResNetConv(in_channels=3, winsize=self.winsize)   # outchannels is 128
        self.c1 = nn.Conv1d(in_channels=128, out_channels=5, kernel_size=8, padding='same')
        self.l1 = nn.LSTM(input_size=5, hidden_size=64, bias=False, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=64, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )
    
    def forward(self, x):
        # x is batch_size x 303, want shape: batch_size x 3 x 101
        x = x.view(-1, 3, self.winsize)

        x = self.resnet(x)
        x = self.c1(x)
        x = nn.functional.relu(x)
        x = torch.transpose(x, 1, 2)
        o, (h,c) = self.l1(x)
        o = nn.functional.relu(o[:,-1,:])
        logits = self.mlp(o)

        return logits

    @staticmethod
    def get_criterion():
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=0.001)