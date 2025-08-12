import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from src.utils import show_image_tensor

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(model, criterion, image, optimizer, num_epochs,patience=500):
    model.train()
    best_model=None
    best_k=None
    count=0
    losses = []
    min_loss=100
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # 每个epoch开始时清零梯度
        
        predicted_aperture, reconstructed_diffraction,k = model(image)
        loss = criterion(reconstructed_diffraction, image)
        
        loss.backward()
        optimizer.step()

        if loss.item()<min_loss:
            min_loss=loss.item()
            best_model=model.state_dict()
            best_k=k
            count=0
        else:
            count=count+1
        
        losses.append(loss.item())
        
       
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        if count>patience:
            break
        """
        if epoch % 10 == 0 and epoch > 0:
                model.anneal_temperature(factor=0.8)
                current_temp = model.gumbel_layer.temperature.item()
                print(f"Epoch {epoch}, Temperature: {current_temp:.3f}")
                show_image_tensor(predicted_aperture)
                show_image_tensor(reconstructed_diffraction)
            # 监控二值化程度
        """
    return losses,best_model,best_k

# 测试函数
import torchvision
def test(model,image,show=True):
    model.eval()
    with torch.no_grad():
        predicted_aperture, reconstructed_diffraction,k= model(image)
        if show:
            print("Original Image:")
            show_image_tensor(image, "Original Diffraction Pattern")
            
            print("Predicted Aperture:")
            show_image_tensor(predicted_aperture, "Predicted Aperture")
            
            print("Reconstructed Diffraction:")
            show_image_tensor(reconstructed_diffraction, "Reconstructed Diffraction Pattern")
            """
            reconstructed_diffraction=reconstructed_diffraction/k

            print(f"Reconstructed Diffraction/{k}:")
            show_image_tensor(reconstructed_diffraction, "Reconstructed Diffraction Pattern")

            a=Fresnal(572,d=0.8,rate=8e-6)
            rec_2=a(predicted_aperture)
            show_image_tensor(reconstructed_diffraction, "Reconstructed Diffraction Pattern")

            #loss=criterion(reconstructed_diffraction,image)
            torchvision.utils.save_image(reconstructed_diffraction,"rec.png",normalize=True)
            torchvision.utils.save_image(image,"orig.png")
            """
    return predicted_aperture,reconstructed_diffraction

def show_MSE_loss_distribution(image,reconstructed_diffraction):
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
    fig.savefig('loss.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



