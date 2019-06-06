import torch
import torch.nn as nn


'''def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model
'''

def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d,
         BN=False, ws=False, activ=nn.LeakyReLU(0.2), gainWS=2):
    convlayer = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=bias)
    layers = []
    if ws:
        layers.append(WScaleLayer(convlayer, gain=gainWS))
    if BN:
        layers.append(nn.BatchNorm2d(nout))
    if activ is not None:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()')
            layers.append(activ(num_parameters=1))
        else:
            # if activ == nn.PReLU(), the parameter will be shared for the whole network !
            layers.append(activ)
    layers.insert(ws, convlayer)
    return nn.Sequential(*layers)
    
class ResidualConv(nn.Module):
    def __init__(self, nin, nout, bias=False, BN=False, ws=False, activ=nn.LeakyReLU(0.2)):
        super(ResidualConv, self).__init__()

        convs = [conv(nin, nout, bias=bias, BN=BN, ws=ws, activ=activ),
                 conv(nout, nout, bias=bias, BN=BN, ws=ws, activ=None)]
        self.convs = nn.Sequential(*convs)

        res = []
        if nin != nout:
            res.append(conv(nin, nout, kernel_size=1, padding=0, bias=False, BN=BN, ws=ws, activ=None))
        self.res = nn.Sequential(*res)

        activation = []
        if activ is not None:
            if activ == nn.PReLU:
                # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()')
                activation.append(activ(num_parameters=1))
            else:
                # if activ == nn.PReLU(), the parameter will be shared for the whole network !
                activation.append(activ)
        self.activation = nn.Sequential(*activation)

    def forward(self, input):
        out = self.convs(input)
        return self.activation(out + self.res(input))
        
        
def upSampleConv_Res(nin, nout, upscale=2, bias=False, BN=False, ws=False, activ=nn.LeakyReLU(0.2)):
    return nn.Sequential(
        nn.Upsample(scale_factor=upscale),
        ResidualConv(nin, nout, bias=bias, BN=BN, ws=ws, activ=activ)
    )
    
    
    
def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1 ):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation ),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model
    
def conv_block_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model

def conv_block_Asym(in_dim, out_dim, kernelSize):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize,1],   padding=tuple([2,0])),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0,2])),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_Asym_Inception(in_dim, out_dim, kernel_size, padding, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size,1],   padding=tuple([padding*dilation,0]), dilation = (dilation,1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0,padding*dilation]), dilation = (dilation,1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model
    

def conv_block_Asym_Inception_WithIncreasedFeatMaps(in_dim, mid_dim, out_dim, kernel_size, padding, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, kernel_size=[kernel_size,1],   padding=tuple([padding*dilation,0]), dilation = (dilation,1)),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0,padding*dilation]), dilation = (dilation,1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model
    
        
def conv_block_Asym_ERFNet(in_dim, out_dim, kernelSize, padding, drop, dilation):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize,1],   padding=tuple([padding,0]), bias = True),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0,padding]), bias = True),
        nn.BatchNorm2d(out_dim, eps=1e-03),
        nn.ReLU(),
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize,1],   padding=tuple([padding*dilation,0]), bias=True, dilation = (dilation,1)),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0,padding*dilation]), bias=True, dilation = (1, dilation)),
        nn.BatchNorm2d(out_dim, eps=1e-03),
        nn.Dropout2d(drop),
    )
    return model
                    
def conv_block_3_3(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model
        
# TODO: Change order of block: BN + Activation + Conv
def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def dilation_conv_block(in_dim,out_dim,act_fn,stride_val,dil_val):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride_val, padding=1, dilation=dil_val),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model
    
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def avrgpool05():
    pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def avrgpool025():
    pool = nn.AvgPool2d(kernel_size=2, stride=4, padding=0)
    return pool


def avrgpool0125():
    pool = nn.AvgPool2d(kernel_size=2, stride=8, padding=0)
    return pool

    
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def maxpool_1_4():
    pool = nn.MaxPool2d(kernel_size=2, stride=4, padding=0)
    return pool

def maxpool_1_8():
    pool = nn.MaxPool2d(kernel_size=2, stride=8, padding=0)
    return pool

def maxpool_1_16():
    pool = nn.MaxPool2d(kernel_size=2, stride=16, padding=0)
    return pool
    
def maxpool_1_32():
    pool = nn.MaxPool2d(kernel_size=2, stride=32, padding=0)
    
    
def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def classificationNet(D_in):
    H = 400
    D_out = 1
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, int(H / 4)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(H / 4), D_out)
    )

    return model
