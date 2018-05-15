import torch
import torch.nn as nn
import numpy as np

    
class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, num_filters, kernel_size, padding, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.batch_norm=batch_norm
        num_hidden_filters = num_filters
        self.conv1 = nn.Conv2d(num_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity(inplace=False)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(num_hidden_filters)
        self.conv2 = nn.Conv2d(num_hidden_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(num_filters)

         
    def forward(self, og_x):
        x = og_x
        x = self.dropout(x)
        x = self.conv1(og_x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        
        out = og_x + x
        if self.batch_norm:
            out = self.batch_norm2(out)
        out = self.nonlinearity(out)
        return out
    
    
    
class DilatedNetwork(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_hidden_features,n_resblocks, dropout=0, padding=1, kernel_size=3,batch_norm=True):
        super(DilatedNetwork, self).__init__()

        self.stages=nn.ModuleList()
        
        num_features=[in_channels]+num_hidden_features

        dilatations = [2**k for k in range(len(num_hidden_features))]
  
        # input convolution block
        block =[] 
        step=0
        #encoder
        for features_in,features_out in [num_features[i:i+2] for i in range(0,len(num_features), 1)][:-1]:
            block +=[nn.Conv2d(features_in, features_out, kernel_size=kernel_size,stride=1, padding=padding)]
            block +=[ResidualBlock(features_out, kernel_size, dilatations[step], dropout=dropout, dilation=dilatations[step],batch_norm=batch_norm)]
            step+=1
        step=0
        #decoder
        dilatations_up=dilatations[::-1]
        dilatations_up=dilatations_up[1:]
        features_up=num_hidden_features[::-1]
        for features_in,features_out in [features_up[i:i+2] for i in range(0,len(num_hidden_features), 1)][:-1]:
            block +=[nn.Conv2d(features_in, features_out, kernel_size=kernel_size,stride=1, padding=padding)]
            block +=[ResidualBlock(features_out, kernel_size, dilatations_up[step], dropout=dropout, dilation=dilatations_up[step],batch_norm=batch_norm)]
            step+=1
            
        block +=[nn.Conv2d(features_up[-1], out_channels, kernel_size=kernel_size,stride=1, padding=padding)]
        self.stages=nn.Sequential(*block)
    
    def forward(self,x):

        for stage in self.stages:
            
            x = stage(x)
            

        return x
    