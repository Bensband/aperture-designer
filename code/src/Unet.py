import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.Fresnal import Fresnal
from src.Fresnal_extra import Fresnal_ex

class Double_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,padding_=0,kernal_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels=out_channels
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=kernal_size,padding=padding_),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=kernal_size,padding=padding_),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self,in_channels,out_channels,padding_=0,kernal_size=3):
        super().__init__()
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool2d(2),
            Double_Conv(in_channels,out_channels,padding_=padding_,kernal_size=kernal_size),
        )

    def forward(self,x):
        return self.maxpool_conv(x)

 
def crop_copy(tens1,tens2):
    delta=(tens1.shape[2]-tens2.shape[2])//2
    return torch.cat([
        tens1[:,:,delta:tens1.shape[2]-delta,delta:tens1.shape[3]-delta],
        tens2],axis=1)


class Up(nn.Module):
    def __init__(self,in_channels,out_channels,conv_pad=0,conv_size=3):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,in_channels//2,2,stride=2,padding=0)
        self.conv=Double_Conv(in_channels,out_channels,padding_=conv_pad,kernal_size=conv_size)

    def forward(self,copy_tens,x):
        out=self.up(x)
        out=crop_copy(copy_tens,out)
        out=self.conv(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels,padding_=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=padding_)

    def forward(self, x):
        return self.conv(x)

# class MiniUnet(nn.Module):
#     def __init__(self,channels=1):
#         super().__init__()
#         self.inc = (Double_Conv(channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.up1 = (Up(256, 128))
#         self.up2 = (Up(128, 64))
#         self.outc = (OutConv(64,channels))

#         self.fresnal=Fresnal(128)

#     def forward(self,x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x2, x3)
#         x = self.up2(x1, x)
#         x = F.interpolate(
#             x,
#             size=(128,128),
#             mode="bilinear",
#             align_corners=False
#         )
#         x = self.outc(x)
#         aperture=F.sigmoid(x)
#         reconstructed_diffraction=self.fresnal(aperture)
#         return aperture,reconstructed_diffraction
    
# class Unet(nn.Module):
#     def __init__(self,channels=1,alpha=1,beta=0,initial_temperature=2.0,d=0.8,rate=60e-6,lambda_=632e-9,is_k=True):
#         super().__init__()
#         self.alpha=alpha
#         self.beta=beta
#         self.inc = (Double_Conv(channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         self.down4 = (Down(512, 1024))
#         self.up1 = (Up(1024, 512))
#         self.up2 = (Up(512, 256))
#         self.up3 = (Up(256, 128))
#         self.up4 = (Up(128, 64))
#         self.outc = (OutConv(64,channels))
#         self.fresnal=Fresnal(572,lambda_=lambda_,d=d,rate=rate)
#         #self.gumbel_layer = GumbelSigmoidLayer(initial_temperature)
         
#     def forward(self,x):
#         mean_input=torch.mean(x)
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x4, x5)
#         x = self.up2(x3, x)
#         x = self.up3(x2, x)
#         x = self.up4(x1, x)
#         x = F.interpolate(
#             x,
#             size=(572,572),
#             mode="bilinear",
#             align_corners=False
#         )
#         x = self.outc(x)
#         aperture=F.sigmoid(self.alpha*(x-self.beta))
#         """
#         aperture=self.gumbel_layer(x,
#                                    hard=False, 
#                                 training_mode=self.training)
#         """
#         reconstructed_diffraction=self.fresnal(aperture)
#         mean_rec=torch.mean(reconstructed_diffraction)
#         k=mean_input/mean_rec
#         reconstructed_diffraction=reconstructed_diffraction*k
#         return aperture,reconstructed_diffraction,k
    
class Unet_512(nn.Module):
    def __init__(self,mode=1,channels=1,alpha=1,beta=0,d=1.58,rate=60e-6,lambda_=632e-9):
        super().__init__()
        self.mode=mode
        self.alpha=alpha
        self.beta=beta
        self.inc = (Double_Conv(channels, 64,padding_=1))
        self.down1 = (Down(64, 128,padding_=1))
        self.down2 = (Down(128, 256,padding_=1))
        self.down3 = (Down(256, 512,padding_=1))
        self.down4 = (Down(512, 1024,padding_=1))
        self.up1 = (Up(1024, 512,conv_pad=1))
        self.up2 = (Up(512, 256,conv_pad=1))
        self.up3 = (Up(256, 128,conv_pad=1))
        self.up4 = (Up(128, 64,conv_pad=1))
        self.outc = (OutConv(64,channels,padding_=0))
        if mode==1:
            self.fresnal=Fresnal(512,d=d,rate=rate,lambda_=lambda_)
        if mode==2:
            self.fresnal=Fresnal_ex(512,d=d,rate=rate,lambda_=lambda_)

    def forward(self,x):
        mean_input=torch.mean(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.outc(x)
        aperture=F.sigmoid(self.alpha*(x-self.beta))
        reconstructed_diffraction=self.fresnal(aperture)
        mean_rec=torch.mean(reconstructed_diffraction)
        k=mean_input/mean_rec
        #k=k*(1/k) #(不想用k优化可以把它去掉注释放到代码里)
        reconstructed_diffraction=reconstructed_diffraction*k
        return aperture,reconstructed_diffraction,k