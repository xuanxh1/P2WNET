"""
图像对叠加可视化工具
使用单应性矩阵将两幅图像进行配准和叠加显示
"""
import torch
import kornia
import kornia.geometry as KG
from torchvision import utils

def overlay_image_pairs(large_img, small_img, H_small_to_large, is_uni_modal=False):
    """
    使用单应性矩阵将小图变换并叠加到大图上，支持灰度图和彩色图
    
    参数:
        large_img: 大图张量 (B, C, H_large, W_large)
        small_img: 小图张量 (B, C, H_small, W_small)
        H_small_to_large: 单应性矩阵 (B, 3, 3)，从小图到大图的变换
        is_uni_modal: 是否为单模态，False时在非重叠区域添加红色遮罩

    返回:
        visualization: RGB可视化结果 (B, 3, H_large, W_large)
    """

    device = large_img.device
    batch_size = large_img.size(0)
    
    # 获取大图和小图的尺寸
    _, _, h_large, w_large = large_img.shape
    _, _, h_small, w_small = small_img.shape
    
    if large_img.shape[1]==1 and small_img.shape[1]==1:
    
        # 将灰度图转换为RGB图像
        large_img = large_img.repeat(1, 3, 1, 1)
        small_img = small_img.repeat(1, 3, 1, 1)
    # 创建小图区域的掩码并将其变换到大图空间
    mask = torch.ones((batch_size, 1, h_small, w_small), device=device)
    transformed_mask = kornia.geometry.transform.warp_perspective(
        mask,
        H_small_to_large,
        (h_large, w_large),
        mode='nearest',
        align_corners=True
    )
    # 如果不是单模态，添加红色遮罩
    if is_uni_modal:
        # 创建红色遮罩
        red_overlay = torch.tensor([1.0, 0.6, 0.2], device=device).view(1, 3, 1, 1)
        red_overlay = red_overlay.expand_as(large_img)
        # 混合结果
        alpha = 0.3
        visualization = large_img * (1 - alpha * (1 - transformed_mask)) + \
                       red_overlay * (alpha * (1 - transformed_mask))
    else:
       
        visualization = large_img.clone()
    # 将变换后的小图叠加到结果上
    warped_small = kornia.geometry.transform.warp_perspective(
        small_img,
        H_small_to_large,
        (h_large, w_large),
        mode='bilinear',
        align_corners=True
    )
    # 将变换后的小图叠加到结果上
    visualization = visualization * (1 - transformed_mask) + warped_small * transformed_mask

    return visualization