
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

class FrequencyDomainLoss(nn.Module):
    def __init__(self, loss_type='weighted', high_freq_weight=2.0):
        super().__init__()
        self.loss_type = loss_type
        self.high_freq_weight = high_freq_weight
    def create_frequency_mask(self,h, w, device):
        """创建频域掩码，用于分离高低频成分"""
        # 创建从中心到边缘的距离矩阵
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(device), x.to(device)
        
        # 计算到中心的距离
        distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        max_distance = torch.sqrt(torch.tensor(center_h**2 + center_w**2))
        
        # 归一化距离 [0, 1]
        normalized_distance = distance / max_distance
        
        return normalized_distance
        
    def weighted_frequency_loss(self, pred, target):
        """加权频域损失：高频成分权重更大"""
        # 进行2D FFT
        pred=pred.squeeze(0).squeeze(0)
        target=target.squeeze(0).squeeze(0)
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # 获取图像尺寸
        h=pred.size()[0]
        w=pred.size()[1]
        
        # 创建频域权重掩码
        freq_mask = self.create_frequency_mask(h, w, pred.device)

        # 高频权重更大，低频权重较小
        # freq_mask从中心(0)到边缘(1)，所以边缘(高频)权重大
        weight_mask = 1.0 + self.high_freq_weight * freq_mask
        
        # 计算加权MSE损失
        diff = torch.abs(pred_fft - target_fft) ** 2
        weighted_loss = torch.mean(diff * weight_mask.unsqueeze(0).unsqueeze(0))
        weighted_loss=weighted_loss/(h*w)
        return weighted_loss
    
    def forward(self, pred, target):
        if self.loss_type == 'weighted':
            return self.weighted_frequency_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
class PSNRLoss(nn.Module):
    """
    PSNR损失函数
    Peak Signal-to-Noise Ratio Loss
    """
    def __init__(self, max_val=1.0, reduction='mean'):
        """
        Args:
            max_val (float): 像素值的最大值，默认1.0（归一化后）
            reduction (str): 'mean', 'sum' 或 'none'
        """
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测图像 [B, C, H, W]
            target (torch.Tensor): 目标图像 [B, C, H, W]
        Returns:
            torch.Tensor: PSNR损失值
        """
        # 计算MSE
        mse = F.mse_loss(pred, target, reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)  # [B]
        
        # 计算PSNR
        psnr = 10 * torch.log10(self.max_val ** 2 / (mse + 1e-8))
        
        # 由于我们要最小化损失，所以返回负的PSNR
        psnr_loss = -psnr
        
        if self.reduction == 'mean':
            return psnr_loss.mean()
        elif self.reduction == 'sum':
            return psnr_loss.sum()
        else:
            return psnr_loss


class SSIMLoss(nn.Module):
    """
    SSIM损失函数
    Structural Similarity Index Loss
    """
    def __init__(self, window_size=11, sigma=1.5, channel=3, reduction='mean'):
        """
        Args:
            window_size (int): 高斯窗口大小
            sigma (float): 高斯窗口标准差
            channel (int): 图像通道数
            reduction (str): 'mean', 'sum' 或 'none'
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.reduction = reduction
        
        # 创建高斯窗口
        self.window = self._create_window(window_size, sigma, channel)
        
    def _gaussian_window(self, window_size, sigma):
        """创建1D高斯窗口"""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g
        
    def _create_window(self, window_size, sigma, channel):
        """创建2D高斯窗口"""
        _1D_window = self._gaussian_window(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """计算SSIM"""
        # 常数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 确保window在正确的设备上
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        
        # 计算均值
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # 计算SSIM
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        
        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
            
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测图像 [B, C, H, W]
            target (torch.Tensor): 目标图像 [B, C, H, W]
        Returns:
            torch.Tensor: SSIM损失值
        """
        batch_size, channel, height, width = pred.size()
        
        # 检查通道数
        if channel != self.channel:
            self.window = self._create_window(self.window_size, self.sigma, channel)
            self.channel = channel
            
        # 计算SSIM
        ssim_value = self._ssim(pred, target, self.window, self.window_size, channel, 
                               size_average=self.reduction == 'mean')
        
        # 返回1-SSIM作为损失（因为SSIM越大越好，损失越小越好）
        return 1 - ssim_value

class com_loss(nn.Module):
    def __init__(self, lambda_l1, lambda_MSE,lambda_Fre,lambda_psnr=0.01,lambda_ssim=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_MSE = lambda_MSE
        self.lambda_Fre = lambda_Fre
        self.psnr=lambda_psnr
        self.ssim=lambda_ssim

    
    def forward(self, generated, target):
        # Use functional versions directly
        l1 = F.l1_loss(generated, target)
        mse = F.mse_loss(generated, target)
        Fre=FrequencyDomainLoss()
        Psnr=PSNRLoss()
        Ssim=SSIMLoss()
        fre=Fre(generated,target)
        psnr=Psnr(generated,target)
        ssim=Ssim(generated,target)
        loss = self.lambda_l1 * l1 + self.lambda_MSE * mse+self.lambda_Fre*fre+self.psnr*psnr+self.ssim*ssim
        return loss