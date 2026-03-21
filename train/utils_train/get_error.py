"""
误差计算工具
计算各类L2误差和MACE指标
"""
import torch


def get_mace(pts_gt, pts_pred, k=4, search_scale=None):
    """
    计算4个角点的平均误差(MACE)
    
    参数:
        pts_gt: 真实点 [bs,k,2]
        pts_pred: 预测点 [bs,k,2]
        k: 点数量
        search_scale: 搜索图缩放因子
    
    返回:
        mace: 平均角点误差 [bs]
    """
    if search_scale!=None:
        pts_gt = pts_gt*search_scale
        pts_pred = pts_pred*search_scale
    bs = pts_gt.shape[0]
    
    # 计算网格划分数
    x_divisions = int(torch.sqrt(torch.tensor(k, dtype=torch.float32)))
    while k % x_divisions != 0:
        x_divisions -= 1
    y_divisions = k // x_divisions
    
    # 4个角点索引
    corner_indices = [
        0,                                # 左上
        x_divisions-1,                    # 右上
        (y_divisions-1) * x_divisions,    # 左下
        k-1                               # 右下
    ]
    
    # 提取角点
    gt_corners = pts_gt[:, corner_indices, :]
    pred_corners = pts_pred[:, corner_indices, :]
    
    # 计算欧氏距离
    corner_errors = torch.norm(gt_corners - pred_corners, dim=2)
    
    # 计算平均误差
    mace = torch.mean(corner_errors, dim=1)
    
    return mace

def get_l2_error_in_and_out(pts_gt, pts_pred, search_size,search_scale=None):
    """
    计算边界内外点的L2误差
    
    参数:
        pts_gt: 真实点 [bs,k,2]
        pts_pred: 预测点 [bs,k,2]
        search_size: 搜索区域尺寸 [h,w]
        search_scale: 搜索图缩放因子
    
    返回:
        l2_errors: 所有点的误差 [bs*k]
        l2_errors_in: 边界内点的误差
        l2_errors_out: 边界外点的误差
    """

    # 计算所有点的L2误差
    if search_scale is not None:
        l2_errors = torch.norm(pts_pred*search_scale - pts_gt*search_scale, dim=-1)
    else:
        l2_errors = torch.norm(pts_pred - pts_gt, dim=-1)
    
    # 展平
    l2_errors = l2_errors.flatten()
    
    # 提取搜索区域尺寸
    h, w = search_size
    
    pts_gt = pts_gt.floor()
    # 检查点是否在边界内
    in_bounds_mask = (pts_gt[..., 0] >= 0) & (pts_gt[..., 0] <= h) & (pts_gt[..., 1] >= 0) & (pts_gt[..., 1] <= w)
    
    in_bounds_mask = in_bounds_mask.flatten()

    # 提取边界内外的误差
    l2_errors_in = l2_errors[in_bounds_mask]
    l2_errors_out = l2_errors[~in_bounds_mask]

    return l2_errors, l2_errors_in, l2_errors_out