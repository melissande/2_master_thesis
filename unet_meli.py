"""
U-net model from https://arxiv.org/abs/1505.04597 implemented with residual blocks, batch norm and dropout 
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
        
class conv_block(nn.Module):
    """
    block composed of  two successive convolutions each followed by batch normalization and ReLU activation function and residual connection for the two output of the convolution layers
    """
    def __init__(self, in_ch, out_ch,dropout):
        super(conv_block, self).__init__()
        
        self.conv1=nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2=nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.BN=nn.BatchNorm2d(out_ch)
        self.ReLU=nn.ReLU(inplace=True)
        self.dropout = dropout
        if self.dropout > 0:
            self.drop = nn.Dropout2d(self.dropout)    
            
    def forward(self, x):
        x_1=self.conv1(x)
        if self.dropout > 0:
            x_1=self.drop(x_1)
        x_1=self.BN(x_1)
        x_2=self.ReLU(x_1)
        x_2=self.conv2(x_2)
        if self.dropout > 0:
            x_2=self.drop(x_2)
        x_2 = self.BN(x_2)
        out=self.ReLU(x_1+x_2)
#         out=self.ReLU(x_2)
        return out
    



class inconv(nn.Module):
    """
    initial convolution for the U-Net model
    """
    def __init__(self, in_ch, out_ch,dropout):
        super(inconv, self).__init__()
        self.conv = conv_block(in_ch, out_ch,dropout)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    """
    last convolution for the U-Net
    """
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    



class down_residual(nn.Module):
    """
    U-Net down block using on residual block
    """
    def __init__(self, in_ch, out_ch, dropout=0):
        super(down_residual, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block( in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
    
class up_residual(nn.Module):
    """
    U-Net up block using one residual block
    """
    def __init__(self, in_ch, out_ch, dropout=0):
        super(up_residual, self).__init__()
        
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch,dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, diffX // 2,diffY // 2, diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        size = m.weight.size()
        fan_row = size[1] # number of rows
        fan_col = size[2] # number of columns
        fan_ch = size[3] # number of channels
        variance = np.sqrt(2.0/(fan_row * fan_col *fan_ch))
        m.weight.data.normal_(0.0, variance)
 


        
class UNet(nn.Module):
    """
    U-Net model with custom number of layers, dropout and batch normalization
    """
    def __init__(self, in_channels, out_channels, depth = 5, n_features_zero = 64, dropout=0,distance_net=False,threshold=20,bins=15):
        """
        initialize the model
        Args:
            in_channels (int): number of input channels (image=9)
            out_channels (int): number of output channels (n_classes)
            depth (int): number of down/up layers
            n_features (int): number of initial features
            dropout (float): float in [0,1]: dropout probability
        """
        super(UNet, self).__init__()
        n_features = n_features_zero
        self.inc = inconv(in_channels,n_features,dropout=dropout)
        # DOWN
        self.downs = torch.nn.ModuleList()
        for k in range(depth):
            d = down_residual(n_features, 2*n_features, dropout=dropout)
            n_features = 2 * n_features
            self.downs += [d]
        # UP
        
        self.ups = torch.nn.ModuleList()
        for k in range(depth):
            u = up_residual(n_features, n_features//2, dropout=dropout)
            n_features = n_features // 2
            self.ups += [u]
        self.outc = outconv(n_features, out_channels)
        self.distance_net=distance_net
        if self.distance_net:
            self.outc2= outconv(n_features, bins)
            
    
        
        
    def forward(self, x):
        
        x = self.inc(x)
        bridges = []
        for d in self.downs:
            bridges += [x]
            x = d(x)
 
        for k,u in enumerate(self.ups):
            x = u(x,bridges[len(bridges)-1-k])
        if self.distance_net:
            x_seg=self.outc(x)
            x_dist=torch.cat((x, x_seg), 1)
            x_dist = self.outc2(x)
            return x_dist,x_seg
        else:
            x = self.outc(x)
            return x
        

