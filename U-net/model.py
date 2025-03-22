import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.transforms.functional import pil_to_tensor

import torch.nn as nn
import torch.nn.functional as F


#Block with 2 succesive convolutions (with same padding in this case to keep the borders)
class ConvBlock(nn.Module):
    #Receives number of input and output channels
    def __init__(self, in_c, out_c):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1) #same
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1) #same
        self.bn = nn.BatchNorm2d(out_c) #Normalize each batch (zero mean->no bias)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #First convolution
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        #Second convolution
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

#Block that performs the pool operation reducing the size in half.
#It also returns the original input to do skip-layer connection
# to the decoder.
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock,self).__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

    
#Block that performs a transpose convolution to upsample the input.
#Use instead of pre-defined interpolation so that parameter
# learning also takes place.
#It also adds a crop of the corresponding encoder output to
# make the skip connection. In this case no cropping is needed,
# as the layers have the same size because of same padding.
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock,self).__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(2*out_c, out_c) #in channels are x2 because of the concatenation
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1) #Concatenate along the channel dimension
        x = self.conv(x)
        return x
    

#Our network will take an rgb image
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        #Encoder
        self.e1 = EncoderBlock(3, 64) #Input rgb image and use 64 filters
        self.e2 = EncoderBlock(64, 128) #reduce image size by half and duplicate no. of filters
        self.e3 = EncoderBlock(128, 256) #reduce image size by half and duplicate no. of filters
        self.e4 = EncoderBlock(256, 512) #reduce image size by half and duplicate no. of filters
        
        #Bottleneck
        self.b = ConvBlock(512, 1024) #Convolution without pooling
        
        #Decoder
        self.d1 = DecoderBlock(1024, 512) #upscale the image and reduce by half the no. of filters
        self.d2 = DecoderBlock(512, 256) #upscale the image and reduce by half the no. of filters
        self.d3 = DecoderBlock(256, 128) #upscale the image and reduce by half the no. of filters
        self.d4 = DecoderBlock(128, 64) #upscale the image and reduce by half the no. of filters
        
        #Classifier
        self.out = nn.Conv2d(64, 1, kernel_size=1, padding=0) #output channels is the number of output classes
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        s4, x = self.e4(x)
        
        x = self.b(x)
        
        x = self.d1(x, s4)
        x = self.d2(x, s3)
        x = self.d3(x, s2)
        x = self.d4(x, s1)
       
        x = self.out(x)
        x = self.sig(x)
        return x
