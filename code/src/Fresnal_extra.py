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
class Fresnal_ex(nn.Module):
    #光阑要求1*1*size*size
    def __init__(self,size,lambda_ = 6320e-10,d = 1.580,rate=60e-6):
        super().__init__()
        self.size=size
        self.L0=size*rate
        self.lambda_=lambda_
        self.d=d
        self.rate=rate
    def forward(self,aperture):
        aperture = aperture.squeeze(0).squeeze(0)
        # 将0-1范围的输入转换为0-2π的相位分布
        phase = aperture * (2 * torch.pi)  # 将[0,1]映射到[0, 2π]
        # 创建相位型光阑的复振幅
        aperture_phase = torch.exp(1j * phase)  # e^(iφ)形式，振幅为1
        k = 2 * np.pi / self.lambda_
        dx = self.rate
        dy = self.rate
        u = torch.fft.fftshift(torch.fft.fftfreq(self.size, dx))
        v = torch.fft.fftshift(torch.fft.fftfreq(self.size, dy))
        kethi, nenta = torch.meshgrid(u, v, indexing='xy')
        # 计算传递函数
        H = torch.exp(1j * k * self.d * (1 - (self.lambda_**2 * (kethi**2 + nenta**2)) / 2))
        H = H.to(DEVICE)
        fa = torch.fft.fftshift(torch.fft.fft2(aperture_phase))
        Fuf = fa * H
        U = torch.fft.ifft2(Fuf)
        I = torch.abs(U)**2
        return I.unsqueeze(0).unsqueeze(0)