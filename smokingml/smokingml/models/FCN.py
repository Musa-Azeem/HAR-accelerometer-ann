from torch import nn
import torch

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First Convolution Block
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.relu1 = nn.ReLU()

        # Second Convolution Block
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.relu2 = nn.ReLU()

        # Third Convolution Block
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()

        # Global Average Pooling
        self.gp = lambda x: torch.mean(x, dim=2)    # Take mean across each feature map (N, C, L) => (N,C)
        
        # Output Later
        self.output = nn.Linear(in_features=128, out_features=1)
    
    def forward(self, x):
        x = x.reshape(-1, 3, 101)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.gp(x)
        logits = self.output(x)

        return logits

    @staticmethod
    def get_criterion():
        return nn.BCEWithLogitsLoss()
    
    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)