import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def bottleNeck(nin,nmid):
    return nn.Sequential(
        nn.BatchNorm2d(nin),
        nn.ReLU(),
        nn.Conv2d(nin,nmid, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(nmid),
        nn.ReLU(),
        nn.Conv2d(nmid,nmid, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(nmid),
        nn.ReLU(),
        nn.Conv2d(nmid,nmid*4, kernel_size=1, stride=1, padding=0),
    )
    
    self.resBlock = nn.Sequential()
    
    def forward(self, input):
        out = self.resBlock(input)
        return out + input
        #return F.leaky_relu(out + input, 0.2)
        

def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation = 1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        #nn.LeakyReLU(0.2)
        nn.PReLU()
    )

def downSampleConv(nin, nout, kernel_size=3, stride=2, padding=1, bias=False):
    return nn.Sequential(
        convBatch(nin,  nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
    )
        
def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        nn.Upsample(scale_factor=upscale),
        convBatch(nin,  nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias),
    )

class residualConv(nn.Module):
    def __init__(self, nin, nout):
        super(residualConv,self).__init__()
        self.convs = nn.Sequential(
            convBatch(nin, nout),
            nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nout)
        )
        self.res = nn.Sequential()
        if nin!=nout:
            self.res = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=1, bias=False),
                nn.BatchNorm2d(nout)
            )

    def forward(self, input):
        out = self.convs(input)
        return F.leaky_relu(out + self.res(input), 0.2)
