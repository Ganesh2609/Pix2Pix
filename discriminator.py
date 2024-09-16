import torch
from torch import nn 



class ConvBlock(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=4, stride:int=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=-0.2)
        )
    
    def forward(self, x):
        return self.conv(x)
    


class Discriminator(nn.Module):
    
    def __init__(self, in_channels:int, features=[63,128,256,512], kernel_size=4):
        
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()
        initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=kernel_size, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.layers.append(initial)
        in_channels = features[0]
        for ch in features[1:-1]:
            self.layers.append(ConvBlock(in_channels=in_channels, out_channels=ch, kernel_size=kernel_size, stride=2, padding=0))
            in_channels = ch
        
        final_feature = ConvBlock(in_channels=in_channels, out_channels=features[-1], kernel_size=kernel_size, stride=1, padding=0)
        self.layers.append(final_feature)
        
        final_conv = nn.Sequential(
            nn.Conv2d(in_channels=features[-1], out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
        self.layers.append(final_conv)
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x