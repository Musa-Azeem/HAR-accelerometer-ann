import torch
from torch import nn

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, winsize):
        super().__init__()
        self.winsize = winsize

        self.resnet_conv = ResNetConv(in_channels=in_channels, winsize=self.winsize)
        # Global Pooling and output
        self.gp = nn.AvgPool1d(kernel_size=self.winsize)    # Take mean across each feature map (N, C, L) => (N,C)
        self.output = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Run convolutional layers of ResNet
        y = self.resnet_conv(x)

        # Run Global pooling and classifier
        y = self.gp(y).squeeze(2)
        logits = self.output(y)
        return logits
    
    @staticmethod
    def get_criterion():
        return ResNetConv.get_criterion()

    @staticmethod
    def get_optimizer(model):
        return ResNetConv.get_optimizer(model)




class ResNetConv(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel_size, use_relu=True):
        if use_relu:
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm1d(num_features=out_channels)
            )
    
    def inner_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=8),
            self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=5),
            self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, use_relu=False)
        )
    
    def shortcut(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels)
        )

    def __init__(self, in_channels, winsize):
        super().__init__()
        self.winsize = winsize

        # First ResNet Block components
        self.shortcut1 = self.shortcut(in_channels=in_channels, out_channels=64)
        self.res1 = self.inner_res_block(in_channels=in_channels, out_channels=64)
        self.relu1 = nn.ReLU()

        # Second Res Block components
        self.shortcut2 = self.shortcut(in_channels=64, out_channels=128)
        self.res2 = self.inner_res_block(in_channels=64, out_channels=128)
        self.relu2 = nn.ReLU()

        # Third Res Block components
        self.shortcut3 = self.shortcut(in_channels=128, out_channels=128)
        self.res3 = self.inner_res_block(in_channels=128, out_channels=128)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # Reshape x: (batch_size, 303) -> (batch_size, 3, 101)
        x = x.view(-1, 3, self.winsize)
        
        # First Res Block
        x_shortcut = self.shortcut1(x)
        h = self.res1(x)
        y = h + x_shortcut
        y = self.relu1(y)

        # Second Res Block
        y_shortcut = self.shortcut2(y)
        h = self.res2(y)
        y = h + y_shortcut
        y = self.relu2(y)

        # Third Res Block
        y_shortcut = self.shortcut3(y)
        h = self.res3(y)
        y = h + y_shortcut
        y = self.relu3(y)

        return y

    @staticmethod
    def get_criterion():
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(
            model.parameters(), 
            lr=0.001, 
            betas=(0.9, 0.999), 
            eps=1e-8
        )