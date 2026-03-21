"""
图像数据增强工具
提供训练和验证阶段的数据预处理和增强方法
"""
import random
from torchvision import transforms

def random_color_jitter(image):
    """
    对图像应用随机颜色抖动
    
    参数:
        image: PIL图像或tensor
    
    返回:
        增强后的图像
    """
    color_jitter = transforms.ColorJitter(
        brightness=random.uniform(0, 0.2),
        contrast=random.uniform(0, 0.2),
        saturation=random.uniform(0, 0.2),
        hue=(-0.1, 0.1)  # 使用元组指定范围
    )
    return color_jitter(image)

def get_train_transform_fn(config):
    """
    根据配置创建训练阶段的数据转换函数
    
    参数:
        config: 配置字典，包含model_name和cnn_activation等参数
    
    返回:
        transform: torchvision.transforms组合对象
    """

    if config['cnn_activation']=='relu':
        transform = transforms.Compose([
        transforms.Lambda(random_color_jitter),  # 随机调整亮度
        transforms.ToTensor(),  # 转换为tensor并归一化至[0,1]
        ])
    else:
        transform = transforms.Compose([
        transforms.Lambda(random_color_jitter),  # 随机调整亮度
        transforms.ToTensor(),  # 转换为tensor并归一化至[0,1]
        transforms.Normalize((0.5,), (0.5,)),  # 归一化到[-1,1]
        ])

    return transform

def get_val_transform_fn(config):
    """
    根据配置创建验证阶段的数据转换函数(不包含数据增强)
    
    参数:
        config: 配置字典，包含model_name和cnn_activation等参数
    
    返回:
        transform: torchvision.transforms组合对象
    """
    
    if config['cnn_activation']=='relu':
        transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor并归一化至[0,1]
        ])
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor并归一化至[0,1]
        transforms.Normalize((0.5,), (0.5,)),  # 归一化到[-1,1]
        ])

    return transform

def val_inverse_transform(image_tensor, config):
    """
    将归一化后的图像tensor还原为原始值域
    
    参数:
        image_tensor: 归一化后的图像tensor
        config: 配置字典
    
    返回:
        image: 反归一化后的图像tensor
    """
    transform = get_val_transform_fn(config) 
    normalize_transform = [t for t in transform.transforms if isinstance(t, transforms.Normalize)][0]
    mean = normalize_transform.mean
    std = normalize_transform.std
    
    image = image_tensor.clone()
    
    # 针对每个样本的每个通道进行处理
    for batch_idx in range(image.shape[0]):  # 遍历batch
        for channel_idx in range(image.shape[1]):  # 遍历channel
            image[batch_idx, channel_idx] = image[batch_idx, channel_idx] * std[channel_idx] + mean[channel_idx]
    
    return image