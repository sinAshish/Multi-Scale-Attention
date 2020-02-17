import torch
from torch import nn

from .resnext import resnext50, resnext101

class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext101()
        
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
