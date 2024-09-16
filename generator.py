    import torch 
    from torch import nn



    class Block(nn.Module):
        
        def __init__(self, in_channels:int, out_channels:int, down=True, act='relu', use_dropout=False):
            super(Block, self).__init__()
            if down:
                self.block = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU() if act=='relu' else nn.LeakyReLU(negative_slope=0.2)
                )
            else:
                self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU() if act == 'relu' else nn.LeakyReLU()
                )
            self.use_dropout = use_dropout
            self.dropout = nn.Dropout(p=0.5)
        
        def forward(self, x):
            x = self.block(x)
            return self.dropout(x) if self.use_dropout else x
        
        
        
    class Generator(nn.Module):
        
        def __init__(self, in_channels:int, out_channels:int, features=[64, 128, 256, 512, 512, 512, 512]):
            
            super(Generator, self).__init__()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            
            initial_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.downs.append(initial_conv)
            
            prev_channels = features[0]
            for curr_channels in features[1:]:
                self.downs.append(Block(in_channels=prev_channels, out_channels=curr_channels, act='leaky_relu', down=True, use_dropout=False))
                prev_channels = curr_channels
                
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=features[-1], out_channels=features[-1], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                nn.ReLU()
            )
            
            features.reverse()
            self.ups.append(Block(in_channels=features[0], out_channels=features[1], act='relu', down=False, use_dropout=True))
            prev_channels = features[1]
            for curr_channels in features[1:3]:
                self.ups.append(Block(in_channels=prev_channels*2, out_channels=curr_channels, act='relu', down=False, use_dropout=True))
                prev_channels = curr_channels
            for curr_channels in features[3:]:
                self.ups.append(Block(in_channels=prev_channels*2, out_channels=curr_channels, act='relu', down=False, use_dropout=False))
                prev_channels = curr_channels
            
            self.final_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=curr_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        
        
        def forward(self, x):
            
            x = self.downs[0](x)
            down_val = []
            for layer in self.downs[1:]:
            x = layer(x)
            down_val.append(x)
            down_val.reverse()
            
            x = self.bottleneck(x)
            
            x = self.ups[0](x)
            idx = 0
            for layer in self.ups[1:]:
                x = torch.cat([x, down_val[idx]], dim=1)
                x = layer(x) 
                idx += 1
            
            x = self.final_conv(x)
            return x
                
                