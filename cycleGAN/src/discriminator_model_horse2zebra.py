import torch 
import torch.nn as nn
import config

class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, std, std_decay_rate):
        super().__init__()
        self.conv = nn.Sequential(
            GaussianNoise(std, std_decay_rate),
            nn.Conv2d(in_channels, out_channels, 4 , stride, 1, bias = True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0,2),
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, std, std_decay_rate, features = [64, 128, 256, 512]):
        super().__init__()
        self. initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0,2),
                                    )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            stride = 1 if feature == features[-1] else 2
            layers.append(Block(in_channels, feature, stride=stride, std=config.GAUSS_STD, std_decay_rate=config.STD_DECAY_RATE))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((5,1,256,256))
    model = Discriminator(in_channels=1)
    preds = model(x)
    print(f'Shape of pred is: {preds.shape} \n')
    print(f'Model: \n {preds}')

if __name__ == "__main__":
    test()
