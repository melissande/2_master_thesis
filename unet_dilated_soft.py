import torch
import torch.nn as nn


    
class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, num_filters, kernel_size, padding, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,batchNormObject=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        num_hidden_filters = num_filters
        self.conv1 = nn.Conv2d(num_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity(inplace=False)
#         self.batch_norm1 = batchNormObject(num_hidden_filters)
        self.batch_norm1 = nn.BatchNorm2d(num_hidden_filters)
        self.conv2 = nn.Conv2d(num_hidden_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.batch_norm2 = nn.BatchNorm2d(num_filters)
#         self.batch_norm2 = batchNormObject(num_filters)

    def forward(self, og_x):
        x = og_x
        x = self.dropout(x)
        x = self.conv1(og_x)
        x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        out = og_x + x
        out = self.batch_norm2(out)
        out = self.nonlinearity(out)
        return out
    
class ConvolutionalEncoder(nn.Module):
    """
    Convolutional Encoder providing skip connections
    """
    def __init__(self,n_features_input,num_hidden_features,kernel_size,padding,n_resblocks,dropout=0.2, blockObject=ResidualBlock,batchNormObject=nn.BatchNorm2d):
        """
        n_features_input (int): number of intput features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        n_resblocks (int): number of residual blocks at each stage
        dropout (float): dropout probability
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalEncoder,self).__init__()
        self.n_features_input = n_features_input
        self.num_hidden_features = num_hidden_features
        self.stages = nn.ModuleList()
        # input convolution block
        block = [nn.Conv2d(n_features_input, num_hidden_features[0], kernel_size=kernel_size,stride=1, padding=padding)]
        
        # layers
        
        for features_in,features_out in [num_hidden_features[i:i+2] for i in range(0,len(num_hidden_features), 1)][:-1]:
              
            for _ in range(n_resblocks):
                block += [blockObject(features_in, kernel_size, padding, dropout=dropout,batchNormObject=batchNormObject)]
            # downsampling   
            self.stages.append(nn.Sequential(*block))
            block = [nn.MaxPool2d(2),nn.Conv2d(features_in, features_out, kernel_size=1,padding=0 ),nn.BatchNorm2d(features_out),nn.ReLU()] #batchNormObject(features_out)
            
        self.stages.append(nn.Sequential(*block))
          
    def forward(self,x):
        skips=[]
        for stage in self.stages:            
            x = stage(x)
            skips.append(x)
        return x,skips
    def getInputShape(self):
        return (-1,self.n_features_input,-1,-1)
    def getOutputShape(self):
        return (-1,self.num_hidden_features[-1], -1,-1)
    
            
class ConvolutionalDecoder(nn.Module):
    """
    Convolutional Decoder taking skip connections
    """
    def __init__(self,n_features_output,num_hidden_features,kernel_size,padding,n_resblocks,dropout,blockObject=ResidualBlock,batchNormObject=nn.BatchNorm2d):
        """
        n_features_output (int): number of output features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        n_resblocks (int): number of residual blocks at each stage
        dropout (float): dropout probability
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalDecoder,self).__init__()
        self.n_features_output = n_features_output
        self.num_hidden_features = num_hidden_features
        self.upConvolutions = nn.ModuleList()
        self.skipMergers = nn.ModuleList()
        self.residualBlocks = nn.ModuleList()
        # input convolution block
        # layers
        for features_in,features_out in [num_hidden_features[i:i+2] for i in range(0,len(num_hidden_features), 1)][:-1]:
            # downsampling
            
            self.upConvolutions.append(nn.Sequential(nn.ConvTranspose2d(features_in, features_out, kernel_size=2, stride=2,padding=0 ),nn.BatchNorm2d(features_out),nn.ReLU()))#batchNormObject(features_out)
            self.skipMergers.append(nn.Conv2d(2*features_out, features_out, kernel_size=kernel_size,stride=1, padding=padding))
            # residual blocks
            block = []
            for _ in range(n_resblocks):
                block += [blockObject(features_out, kernel_size, padding, dropout=dropout,batchNormObject=batchNormObject)]
            self.residualBlocks.append(nn.Sequential(*block))   
        # output convolution block
        block = [nn.Conv2d(num_hidden_features[-1],n_features_output, kernel_size=kernel_size,stride=1, padding=padding)]
        self.output_convolution = nn.Sequential(*block)

    def forward(self,x, skips):
        for up,merge,conv,skip in zip(self.upConvolutions,self.skipMergers, self.residualBlocks,skips):
            x = up(x)
            cat = torch.cat([x,skip],1)
            x = merge(cat)
            x = conv(x)
        return self.output_convolution(x)
    def getInputShape(self):
        return (-1,self.num_hidden_features[0],-1,-1)
    def getOutputShape(self):
        return (-1,self.n_features_output, -1,-1)

    
class DilatedConvolutions2(nn.Module):
    """
    Sequential Dialted convolutions
    """
    def __init__(self, n_channels,dropout,kernel_size,blockObject=ResidualBlock,batchNormObject=nn.BatchNorm2d):
        super(DilatedConvolutions2, self).__init__()
        n_convolutions=3
        self.filters=[n_channels,n_channels,n_channels,n_channels,n_channels*2,n_channels*4,n_channels*4,n_channels]
        self.dilatations = [1,1]+[2**(k+1) for k in range(n_convolutions)]+[1,1]
        self.stages=nn.ModuleList()
        self.downsamples=nn.ModuleList()
        
        for (features_in,features_out),d in zip([self.filters[i:i+2] for i in range(0,len(self.filters), 1)][:-1],self.dilatations):

            self.stages.append(nn.Sequential(nn.Conv2d(features_in, features_out, kernel_size=kernel_size,stride=1, padding=d,dilation=d),nn.BatchNorm2d(features_out),nn.ReLU()))
            if (features_out != n_channels):
                self.downsamples.append(nn.Sequential(nn.Conv2d(features_out, n_channels, kernel_size,stride=1,padding=1),
                    nn.BatchNorm2d(n_channels)))
            else:
                self.downsamples.append(nn.Sequential())
                
            

    def forward(self,x):
        
        skips = []
        for b,d in zip(self.stages,self.downsamples):
            x = b(x)
            if len(d)>0:
                skips.append(d(x))
            else:
                skips.append(x)
        return x, skips
    
class UNet(nn.Module):
    """
    U-Net model with dynamic number of layers, Residual Blocks, Dilated Convolutions, Dropout and Group Normalization
    """
    def __init__(self, in_channels, out_channels, num_hidden_features,n_resblocks,dropout=0,  padding=1, kernel_size=3,group_norm=32):
        """
        initialize the model
        Args:
            in_channels (int): number of input channels (image=3)
            out_channels (int): number of output channels (n_classes)
            num_hidden_features (list(int)): number of hidden features for each layer (the number of layer is the lenght of this list)
            n_resblocks (int): number of residual blocks at each layer 

            dropout (float): float in [0,1]: dropout probability
            padding (int): padding for the convolutions
            kernel_size (int): kernel size for the convolutions
            group_norm (bool): number of groups to use for Group Normalization, default is 32, if zero: use nn.BatchNorm2d
        """
        super(UNet, self).__init__()
        if group_norm > 0:
            for h in num_hidden_features:
                assert h%group_norm==0, "Number of features at each layer must be divisible by 'group_norm'"
        blockObject = ResidualBlock
        batchNormObject = lambda n_features : nn.GroupNorm(group_norm,n_features) if group_norm > 0 else nn.BatchNorm2d
        self.encoder = ConvolutionalEncoder(in_channels,num_hidden_features,kernel_size,padding,n_resblocks,dropout=dropout,blockObject=blockObject,batchNormObject=batchNormObject)

        self.dilatedConvs = DilatedConvolutions2(num_hidden_features[-1],dropout,kernel_size,blockObject=blockObject,batchNormObject=batchNormObject)
        self.decoder = ConvolutionalDecoder(out_channels,num_hidden_features[::-1],kernel_size,padding,n_resblocks,dropout=dropout,blockObject=blockObject,batchNormObject=batchNormObject)
        
    def forward(self, x):
        x,skips = self.encoder(x)
        x,dilated_skips = self.dilatedConvs(x)
        for d in dilated_skips:
            x += d
        x += skips[-1]
        x = self.decoder(x,skips[:-1][::-1])
        return x