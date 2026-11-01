import torch
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np


# 假设我们使用的均值和标准差
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


# mean = torch.tensor([0.5, 0.5, 0.5])
# std = torch.tensor([0.5, 0.5, 0.5])
        
# 定义去归一化函数
def denormalize(tensor, mean, std):
    """将归一化的图像去归一化"""
    # 反向归一化
    denorm_tensor = tensor * std[:, None, None] + mean[:, None, None]
    return denorm_tensor.clamp(0, 1)  # 确保值在 [0, 1] 范围内


# 可视化
def show_image(tensor,id):
    tensor = denormalize(tensor, mean, std)
    """显示图像"""
    plt.imshow(tensor.permute(1, 2, 0).numpy())  # 转换为 (H, W, C)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    # plt.pause(5)
    plt.savefig('vis_imgs/%d.png' % id)

# 显示去归一化后的图像


# if os.path.exists(os.path.join('/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/psaq_generated', 'psaq_gen_deit_tiny.npy')):
#     data = np.load(os.path.join('/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/psaq_generated', 'psaq_gen_deit_tiny.npy'))
#     data = torch.from_numpy(data)


# if os.path.exists(os.path.join('/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/naive_generated', 'gen_deit_tiny.npy')):
#     data = np.load(os.path.join('/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/naive_generated', 'gen_deit_tiny.npy'))
#     data = torch.from_numpy(data)

# path = "/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/generated_imgs/my_generated/gen_vit_base_1024.npy"



# path = "/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/syn_data_64.npy"

path = "/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/generated_imgs/psaq_generated/psaq_gen_deit_small.npy"

# path = "/home/jrq/Desktop/codes/quant_repo/data_free_vit_exp/generated_imgs/naive_generated/gen_deit_tiny.npy"


if os.path.exists(path):
    data = np.load(path)
    data = torch.from_numpy(data)

# denormalized_image = torch.from_numpy(data[0])


# img = torch.randn((3, 224, 224))
# show_image(img, 0)

for i in range(1, 10):
    show_image(data[i], i)