# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

from functools import reduce

import torch

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init



class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))
        
        
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups. def=8
            depth: number of layers.   def=29
            nlabels: number of classes
            base_width: base number of channels in each group. def=64
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)
        
        
class myResNext(nn.Module):
    
    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        resnext_101_32x4d = nn.Sequential(  # Sequential,
    nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                              nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                              nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                              nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.AvgPool2d((7, 7), (1, 1)),
    Lambda(lambda x: x.view(x.size(0), -1)),  # View,
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2048, 1000)),  # Linear,
)

