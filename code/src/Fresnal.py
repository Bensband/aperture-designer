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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Fresnal(nn.Module):
    def __init__(self,size,size_H=512,lambda_ = 6320e-10,d = 1.580,rate=60e-6):
        super().__init__()
        self.size=size
        self.size_H=size_H
        self.L0=size*rate
        self.lambda_=lambda_
        self.d=d
        self.rate=rate
    def forward(self,aperture):
        aperture=aperture.squeeze(0).squeeze(0)
        k = 2 * np.pi / self.lambda_
        dx=self.rate
        dy=self.rate
        u = torch.fft.fftshift(torch.fft.fftfreq(self.size, dx))  # x方向空间频率
        v = torch.fft.fftshift(torch.fft.fftfreq(self.size_H, dy))  # y方向空间频率
        kethi, nenta = torch.meshgrid(u, v,indexing='xy')
        H = torch.exp(1j * k * self.d * (1 - (self.lambda_**2 * (kethi**2 + nenta**2)) / 2))
        H=H.to(DEVICE)
        fa = torch.fft.fftshift(torch.fft.fft2(aperture))  # 孔径场的傅里叶变换
        Fuf = fa * H  # 频域相乘
        U = torch.fft.ifft2(Fuf)  # 逆傅里叶变换得到观察场
        I = torch.abs(U)**2
        return I.unsqueeze(0).unsqueeze(0)