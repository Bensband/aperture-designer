import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import math

from src.Fresnal import Fresnal
from src.loss_func import SSIMLoss,PSNRLoss

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def show_image_tensor(tensor, title="Image"):
    """输入一个张量，输出其对应的图像，进行可视化"""
    if tensor.dim() == 4:
        img = tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
    elif tensor.dim() == 3:
        img = tensor.squeeze(0).cpu().detach().numpy()
    else:
        img = tensor.cpu().detach().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    #plt.colorbar()
    plt.show()

def preprocess_image(image_path_or_url, target_size=512, target_size_H=512,normalize=True):
    """
    将图像预处理为 1×1×target_size×target_size 的张量
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸 (默认512)
        normalize: 是否归一化到[0,1]范围
    
    Returns:
        torch.Tensor
    """
    try:
         
            # 从本地文件加载
        img = Image.open(image_path_or_url)
        
        # 转换为RGB模式（去除透明通道）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        print(f"原始图像尺寸: {img.size}")
        
        # 定义预处理变换
        transform = transforms.Compose([
            transforms.Resize((target_size_H, target_size)),  # 调整尺寸
            transforms.Grayscale(num_output_channels=1),    # 转为灰度图
            transforms.ToTensor(),                          # 转为张量并归一化到[0,1]
        ])
        
        # 应用变换
        img_tensor = transform(img)
        
        # 添加batch维度: (1, 64, 64) -> (1, 1, 64, 64)
        img_tensor = img_tensor.unsqueeze(0)
        
        #print(f"处理后张量形状: {img_tensor.shape}")
        #print(f"张量数值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        return img_tensor
        
    except Exception as e:
        print(f"图像处理出错: {e}")
        return None



def show_MSE_loss_distribution(image,reconstructed_diffraction,path):
    #输入的量要求都是二维的
    loss=(image-reconstructed_diffraction)**2
    if loss.is_cuda:
        loss= loss.cpu()
    error_map=loss.numpy()
    fig = plt.figure(figsize=(5.72, 5.72), dpi=100)  # 1080 pixels = 10.8 inches at 100 DPI
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # Full figure, no margins
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Display with no interpolation
    ax.imshow(error_map, cmap='hot', aspect='equal', interpolation='nearest')
    
    # Save with exact dimensions
    fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def calculate_psnr(pred, target, max_val=1.0):
    """
    计算PSNR值（评估用）
    Args:
        pred (torch.Tensor): 预测图像
        target (torch.Tensor): 目标图像
        max_val (float): 最大像素值
    Returns:
        float: PSNR值
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(max_val ** 2 / mse.item())
    return psnr


def calculate_ssim(pred, target, window_size=11, sigma=1.5):
    """
    计算SSIM值（评估用）
    Args:
        pred (torch.Tensor): 预测图像
        target (torch.Tensor): 目标图像
        window_size (int): 窗口大小
        sigma (float): 高斯窗口标准差
    Returns:
        float: SSIM值
    """
    ssim_loss = SSIMLoss(window_size=window_size, sigma=sigma, 
                        channel=pred.size(1), reduction='mean')
    ssim_value = 1 - ssim_loss(pred, target).item()
    return ssim_value

