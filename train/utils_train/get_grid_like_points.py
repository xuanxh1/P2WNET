"""
网格点生成工具
生成图像上等间距的网格点坐标
"""
import torch
import math

def get_grid_like_points_not_include_margin(width, height, k):
    """
    生成等间距网格点(不包含边界)
    
    参数:
        width: 图像宽度
        height: 图像高度
        k: 网格点总数
    
    返回:
        points: 网格点坐标 [k,2]，格式为(y,x)
    """
    # 计算x和y方向的划分数
    x_divisions = int(math.sqrt(k))
    while k % x_divisions != 0:
        x_divisions -= 1
    y_divisions = k // x_divisions
    
    # 计算间距
    x_spacing = width / (x_divisions + 1)
    y_spacing = height / (y_divisions + 1)
    
    # 生成网格点
    points = []
    for i in range(1, y_divisions + 1):
        for j in range(1, x_divisions + 1):
            y = i * y_spacing
            x = j * x_spacing
            points.append([y, x])
    
    return torch.tensor(points, dtype=torch.float32)


def get_grid_like_points_include_margin(width: int, height: int, k: int) -> torch.Tensor:
    """
    生成等间距网格点(包含边界)，覆盖整个图像范围
    
    参数:
        width: 图像宽度，x范围[0, width-1]
        height: 图像高度，y范围[0, height-1]
        k: 网格点总数
    
    返回:
        points: 网格点坐标 [k,2]，格式为(y,x)
    """
    
    # 计算网格行列数
    x_divisions = int(math.sqrt(k))
    while k % x_divisions != 0:
        x_divisions -= 1
    y_divisions = k // x_divisions

    # 计算间距
    x_spacing = (width - 1) / (x_divisions - 1) if x_divisions > 1 else 0
    y_spacing = (height - 1) / (y_divisions - 1) if y_divisions > 1 else 0

    # 生成网格点
    points = []
    for i in range(y_divisions):    
        for j in range(x_divisions):
            y = round(i * y_spacing)  
            x = round(j * x_spacing)  
            points.append([y, x])    

    return torch.tensor(points, dtype=torch.float32)

