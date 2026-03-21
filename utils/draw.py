"""
图像绘制工具
在图像上绘制角点连线和高亮点，用于可视化配准结果
"""
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import kornia


def draw_lines(image_batch, points_batch, line_color="white", line_width=8):
    """
    在批量图像上绘制角点连线，支持灰度图和RGB图像，自动排序角点
    
    参数:
        image_batch: 图像批次 (B, C, H, W)，C可以是1(灰度)或3(RGB)
        points_batch: 四个角点坐标 (B, 4, 2)，[:,:,0]为纵坐标，[:,:,1]为横坐标
        line_color: 线条颜色，默认白色
        line_width: 线条宽度，默认8
    
    返回:
        绘制了线条的图像批次Tensor
    
    功能:
        自动根据坐标排序确定四个角点的位置
    """
    # 保存处理后的图像
    processed_images = []

    # 遍历每个图像和角点
    for image_tensor, points in zip(image_batch, points_batch):
        # 灰度图转RGB
        if image_tensor.size(0) == 1:
            image_tensor = image_tensor.expand(3, -1, -1)
        
        # 转换为PIL图像
        pil_image = TF.to_pil_image(image_tensor)
        
        # 创建绘图对象
        draw = ImageDraw.Draw(pil_image)

        # 按纵坐标排序四个点
        sorted_points = sorted(points, key=lambda p: p[0])
        # 分为上下两组，再按横坐标排序
        top_points = sorted(sorted_points[:2], key=lambda p: p[1])
        bottom_points = sorted(sorted_points[2:], key=lambda p: p[1])

        # 确定四个角点位置
        top_left = (int(top_points[0][1]), int(top_points[0][0]))
        top_right = (int(top_points[1][1]), int(top_points[1][0]))
        bottom_left = (int(bottom_points[0][1]), int(bottom_points[0][0]))
        bottom_right = (int(bottom_points[1][1]), int(bottom_points[1][0]))

        # 绘制四条边
        draw.line([top_left, top_right], fill=line_color, width=line_width)
        draw.line([top_left, bottom_left], fill=line_color, width=line_width)
        draw.line([bottom_left, bottom_right], fill=line_color, width=line_width)
        draw.line([top_right, bottom_right], fill=line_color, width=line_width)

        # 转换回Tensor
        processed_image = TF.to_tensor(pil_image)
        processed_images.append(processed_image)

    # 堆叠为批次
    return torch.stack(processed_images).to(torch.float32)


def highlight_pts(image, pts, color, radius=5):
    """
    在图像上高亮显示点
    
    参数:
        image: 图像批次
        pts: 点坐标，先h后w
        color: 高亮颜色
        radius: 圆圈半径，默认5
    
    返回:
        高亮后的图像批次
    """
    # 保存处理后的图像
    processed_images = []

    # 遍历每张图像和对应的点
    for image_tensor, points in zip(image, pts):
        # 灰度图转RGB
        if image_tensor.size(0) == 1:
            image_tensor = image_tensor.expand(3, -1, -1)

        # 转换为PIL图像
        pil_image = TF.to_pil_image(image_tensor)

        # 创建绘图对象
        draw = ImageDraw.Draw(pil_image)

        # 在每个点上绘制圆圈
        for point in points:
            y,x = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        # 转换回Tensor
        processed_image = TF.to_tensor(pil_image)
        processed_images.append(processed_image)

    # 堆叠为批次
    return torch.stack(processed_images).to(torch.float32)

def create_checker_mixed_image(imgs_search, imgs_template, H_pred_by_usanc, dev, checker_size=16):
    """
    创建棋盘状混合图像用于可视化配准效果
    
    参数:
        imgs_search: 搜索图像 [B, C, H, W]
        imgs_template: 模板图像 [B, C, H_t, W_t]  
        H_pred_by_usanc: 预测的单应性矩阵 [B, 3, 3]
        dev: 计算设备
        checker_size: 棋盘格尺寸(像素)，默认16
    
    返回:
        mixed_image: 棋盘混合图像 [B, C, H, W]
    
    功能:
        在重叠区域交替显示搜索图像和变换后的模板图像
    """
    # 将模板图像变换到搜索图像坐标系
    template_2_search = kornia.geometry.transform.warp_perspective(
        imgs_template, torch.inverse(H_pred_by_usanc), 
        imgs_search.shape[-2:], mode='bicubic',padding_mode='zeros',
        align_corners=True
    )
    
    # 创建棋盘掩码
    h, w = imgs_search.shape[-2:]
    checker_mask = torch.zeros((h, w), device=dev)
    
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            # 创建棋盘模式: 索引和为偶数时mask为1
            if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                checker_mask[i:min(i+checker_size, h), j:min(j+checker_size, w)] = 1
    
    # 扩展mask到所有批次和通道
    checker_mask = checker_mask.unsqueeze(0).unsqueeze(0).expand(imgs_search.shape[0], imgs_search.shape[1], -1, -1)
    
    # 检测模板的有效区域
    template_valid_mask = (template_2_search.abs().sum(dim=1, keepdim=True) > 0.01).float()
    
    # 创建混合图像: 重叠区域应用棋盘模式
    mixed_image = imgs_search.clone()
    mixed_image = mixed_image * (1 - template_valid_mask * (1 - checker_mask)) + \
                 template_2_search * template_valid_mask * (1 - checker_mask)
    
    return mixed_image

def create_template_replaced_image(imgs_search, imgs_template, H_pred_by_usanc, dev):
    """
    创建模板区域被完全替换的图像
    
    参数:
        imgs_search: 搜索图像 [B, C, H, W]
        imgs_template: 模板图像 [B, C, H_t, W_t]  
        H_pred_by_usanc: 预测的单应性矩阵 [B, 3, 3]
        dev: 计算设备
    
    返回:
        replaced_image: 模板区域被替换的图像 [B, C, H, W]
    
    功能:
        在有效区域用变换后的模板图像完全替换搜索图像
    """
    # 将模板图像变换到搜索图像坐标系
    template_2_search = kornia.geometry.transform.warp_perspective(
        imgs_template, torch.inverse(H_pred_by_usanc), 
        imgs_search.shape[-2:], mode='bicubic',padding_mode='border',align_corners=True
    )
    
    # 创建模板的有效区域掩码
    # 先创建全1掩码，然后应用相同的变换
    template_mask = torch.ones_like(imgs_template[:, :1, :, :])
    template_mask_transformed = kornia.geometry.transform.warp_perspective(
        template_mask, torch.inverse(H_pred_by_usanc), 
        imgs_search.shape[-2:], mode='bicubic',padding_mode='zeros',align_corners=True
    )
    
    # 只在高置信度区域替换，避免插值边界
    template_valid_mask = (template_mask_transformed > 0.8).float()
    
    # 创建替换图像
    replaced_image = imgs_search * (1 - template_valid_mask) + template_2_search * template_valid_mask
    
    return replaced_image