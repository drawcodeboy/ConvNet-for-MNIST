import torch
from torch import nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.acti = nn.GELU()
    
    def forward(self, x):
        return self.acti(self.bn(self.conv(x)))

class ConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        self.conv_blk1 = ConvBlock(in_channels, 32)
        self.conv_blk2 = ConvBlock(32, 64)
        
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.l1 = nn.Linear(3136, 100)
        self.l2 = nn.Linear(100, 10)
        self.acti = nn.GELU()
        
        # Apply Weight Initialization Recursively
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # Weight initialization using Xavier Uniform
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.conv_blk1(x)
        x = self.pool(x)
        
        x = self.conv_blk2(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        
        x = self.acti(self.l1(x))
        x = self.l2(x)
        
        return x
