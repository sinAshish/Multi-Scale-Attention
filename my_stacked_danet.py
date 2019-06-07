import torch
import torch.nn.functional as F
from torch import nn

#from resnext import ResNeXt101
import torch
from torch import nn

#import resnext_101_32x4d_
#from config import resnext_101_32_path
from functools import reduce

import torch
import torch.nn as nn
from torch.autograd import Variable
from attention import PAM_Module,CAM_Module,semanticModule
import pdb




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

class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d
        #net.load_state_dict(torch.load(resnext_101_32_path))

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

class DAF_stack(nn.Module):
    def __init__(self):
        super(DAF_stack, self).__init__()
        self.resnext = ResNeXt101()

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        inter_channels = 64
        out_channels=64

        self.conv6_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d())
        self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d()),nn.Softmax2d())
        self.conv6_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d()),nn.Softmax2d())
        self.conv6_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d()),nn.Softmax2d())

        self.conv7_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d()),nn.Softmax2d())
        self.conv7_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d()),nn.Softmax2d())
        self.conv7_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d()),nn.Softmax2d())
        self.conv7_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))#,nn.Softmax2d()),nn.Softmax2d())

        self.conv8_1=nn.Conv2d(64,64,1)
        self.conv8_2=nn.Conv2d(64,64,1)
        self.conv8_3=nn.Conv2d(64,64,1)
        self.conv8_4=nn.Conv2d(64,64,1)
        self.conv8_11=nn.Conv2d(64,64,1)
        self.conv8_12=nn.Conv2d(64,64,1)
        self.conv8_13=nn.Conv2d(64,64,1)
        self.conv8_14=nn.Conv2d(64,64,1)

        self.softmax_1 = nn.Softmax(dim=-1)
        self.pam_attention_1_1=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_1_1=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)

        self.semanticModule_1_1 = semanticModule(128)
        #self.semanticModule_1_2 = semanticModule(128)
        #self.semanticModule_1_3 = semanticModule(128)
        #self.semanticModule_1_4 = semanticModule(128)

        self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
	
	#Dual Attention mechanism 
        self.pam_attention_1_2=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_1_2=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)

        self.pam_attention_1_3=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_1_3=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)

        self.pam_attention_1_4=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_1_4=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)



        self.pam_attention_2_1=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_2_1=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)

        self.semanticModule_2_1 = semanticModule(128)
        #self.semanticModule_2_2 = semanticModule(128)
        #self.semanticModule_2_3 = semanticModule(128)
        #self.semanticModule_2_4 = semanticModule(128)

        self.conv_sem_2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.pam_attention_2_2=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_2_2=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)

        self.pam_attention_2_3=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_2_3=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)

        self.pam_attention_2_4=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			PAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
		
        self.cam_attention_2_4=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),
			CAM_Module(64),
			nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.PReLU()
		)
        

        self.fuse1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.attention4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
            
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
            
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
            
        )
        self.attention1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
            
        )

        self.refine4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.predict4 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 5, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, x):
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        semVector_1_1,semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1),1))


        attn_pam4 = self.pam_attention_1_4(torch.cat((down4, fuse1), 1))
        attn_cam4 = self.cam_attention_1_4(torch.cat((down4, fuse1), 1))

        attention1_4=self.conv8_1((attn_cam4+attn_pam4)*self.conv_sem_1_1(semanticModule_1_1))

        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
        attn_pam3 = self.pam_attention_1_3(torch.cat((down3, fuse1), 1))
        attn_cam3 = self.cam_attention_1_3(torch.cat((down3, fuse1), 1))
        attention1_3=self.conv8_2((attn_cam3+attn_pam3)*self.conv_sem_1_2(semanticModule_1_2))

        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
        attn_pam2 = self.pam_attention_1_2(torch.cat((down2, fuse1), 1))
        attn_cam2 = self.cam_attention_1_2(torch.cat((down2, fuse1), 1))
        attention1_2=self.conv8_3((attn_cam2+attn_pam2)*self.conv_sem_1_3(semanticModule_1_3))

        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
        attn_pam1 = self.pam_attention_1_1(torch.cat((down1, fuse1), 1))
        attn_cam1 = self.cam_attention_1_1(torch.cat((down1, fuse1), 1))
        attention1_1=self.conv8_4((attn_cam1+attn_pam1)*self.conv_sem_1_4(semanticModule_1_4))
        
        
        ##new design with stacked attention

        semVector_2_1, semanticModule_2_1 = self.semanticModule_2_1(torch.cat((down4, attention1_4*fuse1), 1))

        refine4_1=self.pam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4_2=self.cam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4=self.conv8_11((refine4_1+refine4_2)*self.conv_sem_2_1(semanticModule_2_1))

        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(torch.cat((down3, attention1_3*fuse1), 1))
        refine3_1=self.pam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3_2=self.cam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3=self.conv8_12((refine3_1+refine3_2)*self.conv_sem_2_2(semanticModule_2_2))

        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, attention1_2*fuse1), 1))
        refine2_1=self.pam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2_2=self.cam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2=self.conv8_13((refine2_1+refine2_2)*self.conv_sem_2_3(semanticModule_2_3))

        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, attention1_1*fuse1), 1))
        refine1_1=self.pam_attention_2_1(torch.cat((down1,attention1_1*fuse1),1))
        refine1_2=self.cam_attention_2_1(torch.cat((down1,attention1_1*fuse1),1))

        refine1=self.conv8_14((refine1_1+refine1_2)*self.conv_sem_2_4(semanticModule_2_4))
        
        
        
        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')
        
        if self.training:
            return semVector_1_1,\
                   semVector_2_1, \
                   semVector_1_2, \
                   semVector_2_2, \
                   semVector_1_3, \
                   semVector_2_3, \
                   semVector_1_4, \
                   semVector_2_4, \
                   torch.cat((down1, fuse1), 1),\
                   torch.cat((down2, fuse1), 1),\
                   torch.cat((down3, fuse1), 1),\
                   torch.cat((down4, fuse1), 1), \
                   torch.cat((down1, attention1_1*fuse1), 1), \
                   torch.cat((down2, attention1_2*fuse1), 1), \
                   torch.cat((down3, attention1_3*fuse1), 1), \
                   torch.cat((down4, attention1_4*fuse1), 1), \
                   semanticModule_1_4, \
                   semanticModule_1_3, \
                   semanticModule_1_2, \
                   semanticModule_1_1, \
                   semanticModule_2_4, \
                   semanticModule_2_3, \
                   semanticModule_2_2, \
                   semanticModule_2_1, \
                   predict1, \
                   predict2, \
                   predict3, \
                   predict4, \
                   predict1_2, \
                   predict2_2, \
                   predict3_2, \
                   predict4_2
        else:
            return ((predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4)
        
