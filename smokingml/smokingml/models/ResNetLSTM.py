import torch
from torch import nn

class ResNetLSTM(nn.Module):
    def __init__(self, winsize: int):
        super().__init__()
        self.winsize = winsize

        out_channels = 5
        self.resnet = ResNetMod(in_channels=3, out_channels=out_channels, winsize=self.winsize)
        
        hidden_size = 4
        self.l1 = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, bias=False, batch_first=True)
        
        nhl = 192
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=nhl),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(in_features=nhl, out_features=1)
        )
    
    def forward(self, x):
        # x is batch_size x 303, want shape: batch_size x 3 x 101
        x = x.view(-1, 3, self.winsize)

        x = self.resnet(x)

        x = torch.transpose(x, 1, 2)
        o, (h,c) = self.l1(x)
        o = o[:,-1,:]

        logits = self.mlp(o)

        return logits

    @staticmethod
    def get_criterion():
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=3e-4)

class ResNetMod(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel_size, use_relu=True):
        conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.Dropout(p=0.15),
            nn.BatchNorm1d(num_features=out_channels),
        )
        if use_relu:
                conv_block.append(nn.ReLU())

        return conv_block

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

    def __init__(self, in_channels, out_channels, winsize):
        super().__init__()
        self.winsize = winsize
        self.out_channels = out_channels
        nfilters = (2, 2, self.out_channels)

        # First ResNet Block components
        self.shortcut1 = self.shortcut(in_channels=in_channels, out_channels=nfilters[0])
        self.res1 = self.inner_res_block(in_channels=in_channels, out_channels=nfilters[0])
        self.relu1 = nn.ReLU()

        # Second Res Block components
        self.shortcut2 = self.shortcut(in_channels=nfilters[0], out_channels=nfilters[1])
        self.res2 = self.inner_res_block(in_channels=nfilters[0], out_channels=nfilters[1])
        self.relu2 = nn.ReLU()

        # Third Res Block components
        self.shortcut3 = self.shortcut(in_channels=nfilters[1], out_channels=nfilters[2])
        self.res3 = self.inner_res_block(in_channels=nfilters[1], out_channels=nfilters[2])
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


# # model used for 'first_real_lstm_test'
# class ResNetLSTM(nn.Module):
#     def __init__(self, winsize: int):
#         super().__init__()
#         self.winsize = winsize

#         out_channels = 5
#         self.resnet = ResNetMod(in_channels=3, out_channels=out_channels, winsize=self.winsize)
        
#         hidden_size = 4
#         self.l1 = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, bias=False, batch_first=True)
        
#         nhl = 192
#         self.mlp = nn.Sequential(
#             nn.Linear(in_features=hidden_size, out_features=nhl),
#             nn.Dropout(p=0.05),
#             nn.ReLU(),
#             nn.Linear(in_features=nhl, out_features=1)
#         )
    
#     def forward(self, x):
#         # x is batch_size x 303, want shape: batch_size x 3 x 101
#         x = x.view(-1, 3, self.winsize)

#         x = self.resnet(x)

#         x = torch.transpose(x, 1, 2)
#         o, (h,c) = self.l1(x)
#         o = o[:,-1,:]

#         logits = self.mlp(o)

#         return logits

#     @staticmethod
#     def get_criterion():
#         return nn.BCEWithLogitsLoss()

#     @staticmethod
#     def get_optimizer(model):
#         return torch.optim.Adam(model.parameters(), lr=3e-4)

# class ResNetMod(nn.Module):
#     def conv_block(self, in_channels, out_channels, kernel_size, use_relu=True):
#         conv_block = nn.Sequential(
#             nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
#             nn.Dropout(p=0.15),
#             nn.BatchNorm1d(num_features=out_channels),
#         )
#         if use_relu:
#                 conv_block.append(nn.ReLU())

#         return conv_block

#     def inner_res_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             self.conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=8),
#             self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=5),
#             self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, use_relu=False)
#         )
    
#     def shortcut(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
#             nn.BatchNorm1d(num_features=out_channels)
#         )

#     def __init__(self, in_channels, out_channels, winsize):
#         super().__init__()
#         self.winsize = winsize
#         self.out_channels = out_channels
#         nfilters = (2, 2, self.out_channels)

#         # First ResNet Block components
#         self.shortcut1 = self.shortcut(in_channels=in_channels, out_channels=nfilters[0])
#         self.res1 = self.inner_res_block(in_channels=in_channels, out_channels=nfilters[0])
#         self.relu1 = nn.ReLU()

#         # Second Res Block components
#         self.shortcut2 = self.shortcut(in_channels=nfilters[0], out_channels=nfilters[1])
#         self.res2 = self.inner_res_block(in_channels=nfilters[0], out_channels=nfilters[1])
#         self.relu2 = nn.ReLU()

#         # Third Res Block components
#         self.shortcut3 = self.shortcut(in_channels=nfilters[1], out_channels=nfilters[2])
#         self.res3 = self.inner_res_block(in_channels=nfilters[1], out_channels=nfilters[2])
#         self.relu3 = nn.ReLU()

#     def forward(self, x):
#         # Reshape x: (batch_size, 303) -> (batch_size, 3, 101)
#         x = x.view(-1, 3, self.winsize)
        
#         # First Res Block
#         x_shortcut = self.shortcut1(x)
#         h = self.res1(x)
#         y = h + x_shortcut
#         y = self.relu1(y)

#         # Second Res Block
#         y_shortcut = self.shortcut2(y)
#         h = self.res2(y)
#         y = h + y_shortcut
#         y = self.relu2(y)

#         # Third Res Block
#         y_shortcut = self.shortcut3(y)
#         h = self.res3(y)
#         y = h + y_shortcut
#         y = self.relu3(y)

#         return y