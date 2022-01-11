
import torch.nn as nn

num_classes = 2

# Create CNN Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()    
        self.conv_layer1 = self._conv_layer_set(1, 32, 3 , 1)
        self.conv_layer2 = self._conv_layer_set(32, 64, 3, 2)
        self.conv_layer3 = self._conv_layer_set(64, 128, 3, 1)
        self.conv_layer4 = self._conv_layer_set(128, 256, 3, 2)
        self.conv_layer5 = self._conv_layer_set(256, 512, 5, 1)
        self.conv_layer6 = self._conv_layer_set(512, 1024, 3, 2)
  
        self.fc1 = nn.Linear(4415488, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.out = nn.Sigmoid()
        self.batch=nn.LayerNorm(128, elementwise_affine = True) 
        self.drop=nn.Dropout(p=0.15)        
    
    def _conv_layer_set(self, in_c, out_c, ks, strides, batch=False):
        conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(ks, ks, ks), padding=0, stride=strides),
                nn.ReLU(),
            )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
      
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out