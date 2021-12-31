import torch 
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.conv import Conv2d 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, use_act =  True, **kwargs):
        super().__init__()

        if down:
            layers = [nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)]
        else:
            layers = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)]
            #layers = [nn.ConvTranspose2d(in_channels, out_channels, **kwargs)]
        self.conv = nn.Sequential(
            *layers,
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True) if use_act else nn.Identity() # Identity just passes through and doesn't do anything to input
        )
    
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size = 3, padding = 1),
            ConvBlock(channels, channels, use_act = False, kernel_size = 3, padding =1)
         )
    def forward(self, x):
        return x + self.block(x) # This can be done because the input size does not change 
    
class Generator(nn.Module):
    def __init__(self, img_channels, num_features = 64, num_residuals=9): # Note num_residuals can be set to 6 if image is smaller than 256x256
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=6, stride = 1, padding = 3, padding_mode = 'reflect'),
            nn.ReLU(inplace = True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features    , num_features * 2, down = True, kernel_size = 3, stride = 2, padding = 1),
                ConvBlock(num_features * 2, num_features * 4, down = True, kernel_size = 3, stride = 2, padding = 1),
            ]
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down = False, kernel_size = 4, stride = 1, padding = 2),
                ConvBlock(num_features * 2, num_features    , down = False, kernel_size = 4, stride = 1, padding = 1),
            ]
        )

        self.last = nn.Conv2d(num_features * 1, img_channels, kernel_size = 8, stride = 1, padding = 1, padding_mode='reflect')

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

def test():
    img_channel = 3
    img_size = 256
    x = torch.randn((1, img_channel, img_size, img_size))
    gen = Generator(img_channel, 9)
    print(gen)
    print(gen(x).shape)

if __name__ == '__main__':
    test()