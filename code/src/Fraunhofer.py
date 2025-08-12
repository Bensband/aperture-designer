import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
# from src.utils import show_image_tensor
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib.colors import LogNorm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Fraunhofer(nn.Module):
    def __init__(self,lamb,a,z,size):
        super().__init__()
        self.size=size
        self.a=a
        self.z=z
        self.lamb=lamb
    
    def forward(self,aper):
        aper=aper.squeeze(0).squeeze(0)
        x = torch.linspace(-self.size/2, self.size/2, self.size)
        y = torch.linspace(-self.size/2, self.size/2, self.size)
        X, Y = torch.meshgrid(x, y,indexing='xy')

        # 修正相位因子：移除导致位移的线性相位
        kx = (2 * np.pi / self.lamb / self.a) * X
        ky = (2 * np.pi / self.lamb / self.a) * Y
        phase_factor = torch.exp(-1j * (kx * 0 + ky * 0))  # 将相位因子设为1（移除线性相位）
        phase_factor=phase_factor.to(DEVICE)
        # 傅里叶变换
        fourier_transform = torch.fft.fft2(aper * phase_factor)
        fourier_transform_shifted = torch.fft.fftshift(fourier_transform)  # 使用fftshift将零频移到中心
        diffraction_pattern = torch.abs(fourier_transform_shifted) ** 2
        I = diffraction_pattern / torch.max(diffraction_pattern)
        return I.unsqueeze(0).unsqueeze(0)