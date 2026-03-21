"""
点坐标获取和转换工具
用于生成和转换网格点、角点等坐标
"""
from .get_grid_like_points import get_grid_like_points_include_margin,get_grid_like_points_not_include_margin
import torch
from models.utils.warpping_ultis import restore_single_point,restore_multiple_points
import kornia
import cv2
import numpy as np


def get_l2_error_grid_like_k_pts_not_include_margin(tl_gt,size_template,pts_pred,H_matrix,k):
    """
    计算网格点L2误差(不包含边界点)
    
    参数:
        tl_gt: 左上角坐标 [bs,2]
        size_template: 模板尺寸 [h,w]
        pts_pred: 预测点 [bs,k,2]
        H_matrix: 单应性矩阵 [bs,3,3]
        k: 点数量
    
    返回:
        mean_l2_error: 平均L2误差 [bs]
        mean_min_4_l2_error: 最小4个点的平均误差 [bs]
    """
    pts_gt,_ = get_grid_like_points_not_include_margin(tl_gt,size_template,H_matrix,k)
    l2_errors = torch.norm(pts_pred - pts_gt, dim=-1)
    mean_l2_error = torch.mean(l2_errors, dim=1)

    top_4_errors, _ = torch.topk(l2_errors, 4, largest=False, dim=1)
    mean_min_4_l2_error = torch.mean(top_4_errors, dim=1)
    return mean_l2_error, mean_min_4_l2_error

def get_l2_error_grid_like_k_pts_include_margin(tl_gt,size_template,pts_pred,H_matrix,k):
    """
    计算网格点L2误差(包含边界点)
    
    参数:
        tl_gt: 左上角坐标 [bs,2]
        size_template: 模板尺寸 [h,w]
        pts_pred: 预测点 [bs,k,2]
        H_matrix: 单应性矩阵 [bs,3,3]
        k: 点数量
    
    返回:
        mean_l2_error: 平均L2误差 [bs]
        mean_min_4_l2_error: 最小4个点的平均误差 [bs]
    """
    pts_gt,_ = get_grid_like_points_include_margin(tl_gt,size_template,H_matrix,k)
    l2_errors = torch.norm(pts_pred - pts_gt, dim=-1)
    mean_l2_error = torch.mean(l2_errors, dim=1)

    top_4_errors, _ = torch.topk(l2_errors, 4, largest=False, dim=1)
    mean_min_4_l2_error = torch.mean(top_4_errors, dim=1)
    return mean_l2_error, mean_min_4_l2_error

def get_4_pts_in_template_and_search_img(tl_gt,size_template,H_matrix,inner_dis,scale_factor=1):
    """
    获取模板和搜索图中的4个角点(带内边距)
    
    参数:
        tl_gt: 左上角坐标 [bs,2]
        size_template: 模板尺寸 [h,w]
        H_matrix: 单应性矩阵 [bs,3,3]
        inner_dis: 内边距
        scale_factor: 缩放因子
    
    返回:
        pts: 搜索图中的4个点 [bs,4,2]
        temp_pts: 模板图中的4个点 [bs,4,2]
    """
    inner_dis //= scale_factor
    sar_h, sar_w = size_template[0] // scale_factor , size_template[1] // scale_factor
    tr_gt = torch.stack([tl_gt[:,0]+inner_dis, tl_gt[:,1] + sar_w-1-inner_dis], dim = 1) #[bs,2]
    bl_gt = torch.stack([tl_gt[:,0]+sar_h-1-inner_dis, tl_gt[:,1] + inner_dis], dim = 1) #[bs,2]
    br_gt = torch.stack([tl_gt[:,0]+sar_h-1-inner_dis, tl_gt[:,1] + sar_w-1-inner_dis], dim = 1) #[bs,2]
    tl_gt = torch.stack([tl_gt[:,0]+inner_dis, tl_gt[:,1] + inner_dis], dim = 1)
    temp_pts = torch.stack([tl_gt,tr_gt,bl_gt,br_gt],dim=1) #[bs,4,2]

    tl_gt = restore_single_point(tl_gt,H_matrix)
    bl_gt = restore_single_point(bl_gt,H_matrix)
    tr_gt = restore_single_point(tr_gt,H_matrix)
    br_gt = restore_single_point(br_gt,H_matrix)
    
    pts = torch.stack([tl_gt,tr_gt,bl_gt,br_gt],dim=1) #[bs,4,2]

    return pts,temp_pts

def get_4_corner(template_size,dev,bs):
    """
    获取模板图中4个角点坐标(先w后h格式)
    
    参数:
        template_size: 模板尺寸 (h,w)
        dev: 设备
        bs: 批次大小
    
    返回:
        four_corner_in_template: 4个角点 [bs,4,2]
    """
    h,w = template_size
    four_corner_in_template = torch.tensor([[0, 0],
                                        [w-1, 0],
                                        [0, h-1],
                                        [w-1, h-1]], dtype=torch.float32, device=dev)
    four_corner_in_template = four_corner_in_template.repeat(bs,1,1)

    return four_corner_in_template

def get_pred_4_corner_in_search_img_by_gt_H(pts_org_in_template,template_img_size,gt_H_search_to_template):
    """
    用真实单应性矩阵变换4个角点
    
    参数:
        pts_org_in_template: 模板中原始点 [bs,k,2]
        template_img_size: 模板尺寸 (h,w)
        gt_H_search_to_template: 真实单应性矩阵 [bs,3,3]
    
    返回:
        变换后的4个角点 [bs,4,2]
    """
   
    h,w = template_img_size
    b = pts_org_in_template.shape[0]
    four_corner_in_template = torch.tensor([[0, 0],
                                        [w-1, 0],
                                        [0, h-1],
                                        [w-1, h-1]], dtype=pts_org_in_template.dtype, device=pts_org_in_template.device)
    
    four_corner_in_template = four_corner_in_template.repeat(b,1,1)

    four_corner_by_gt_H = kornia.geometry.transform_points(torch.inverse(gt_H_search_to_template),four_corner_in_template)

    return four_corner_by_gt_H[:,:,[1,0]]

def transform_points_with_homography_hw(points, H):
    """
    用单应性矩阵变换(h,w)格式的点
    
    参数:
        points: 点坐标 [B,N,2]，格式为(h,w)
        H: 单应性矩阵 [B,3,3]，基于(w,h)坐标系
    
    返回:
        变换后的点 [B,N,2]，格式为(h,w)
    """
    B, N, _ = points.shape

    points = points[:,:,[1,0]]
    # 转为齐次坐标
    ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
    points_homogeneous = torch.cat([points, ones], dim=-1)
    
    # 转置以便矩阵乘法
    points_homogeneous = points_homogeneous.transpose(1, 2)
    
    # 应用单应性变换
    transformed_points = torch.bmm(H, points_homogeneous)
    
    # 转置回来
    transformed_points = transformed_points.transpose(1, 2)
    
    # 转为非齐次坐标
    denominator = transformed_points[:, :, 2:3]
    transformed_points_2d = transformed_points[:, :, :2] / (denominator+1e-10)
    
    return transformed_points_2d[:,:,[1,0]]

def select_homography_points(template_pts, pts_gt, dst_pts, min_inliers=8, ransac_thresh=1.5, max_iter=2000, max_thresh=5.0):
    """
    用RANSAC筛选单应性变换的内点
    
    参数:
        template_pts: 源图像点 [bs,n,2]
        pts_gt: 真实点 [bs,n,2]
        dst_pts: 目标图像点 [bs,n,2]
        min_inliers: 最少内点数
        ransac_thresh: RANSAC阈值
        max_iter: 最大迭代次数
        max_thresh: 最大阈值
    
    返回:
        selected_pts_gt: 选中的真实点 [bs,n,2]
        selected_dst: 选中的目标点 [bs,n,2]
        selected_src: 选中的源点 [bs,n,2]
        masks: 每个batch的mask列表
    """
    assert template_pts.shape == dst_pts.shape, f"Shape mismatch: template_pts.shape = {template_pts.shape}, dst_pts.shape = {dst_pts.shape}"
    bs, n, _ = template_pts.shape
    device = template_pts.device
    
    # 转为numpy
    src_np = template_pts.cpu().numpy()
    dst_np = dst_pts.cpu().numpy()
    
    selected_src = torch.zeros_like(template_pts)
    selected_dst = torch.zeros_like(dst_pts)
    selected_pts_gt = torch.zeros_like(pts_gt)
    masks = []
    
    for i in range(bs):
        # 转为(x,y)格式
        src_xy = src_np[i, :, ::-1]  
        dst_xy = dst_np[i, :, ::-1]
        
        # 逐步增加阈值找内点
        curr_thresh = ransac_thresh
        best_mask = None
        best_inliers = 0
        
        while curr_thresh <= max_thresh:
            H, mask = cv2.findHomography(
                src_xy, dst_xy,
                method=cv2.USAC_DEFAULT,
                ransacReprojThreshold=curr_thresh,
                maxIters=max_iter,
                confidence=0.999
            )
            
            if H is not None:
                mask = mask.ravel().astype(bool)
                num_inliers = mask.sum()
                
                if num_inliers >= min_inliers:
                    best_mask = mask
                    break
                elif num_inliers > best_inliers:
                    best_mask = mask
                    best_inliers = num_inliers
            
            curr_thresh += 0.5
            
        if best_mask is None:
            # 未找到内点，返回所有点
            best_mask = np.ones(n, dtype=bool)
            
        masks.append(best_mask)
        
        # 保存选中的点
        selected_pts_gt[i][best_mask] = pts_gt[i][best_mask]
        selected_src[i][best_mask] = template_pts[i][best_mask]
        selected_dst[i][best_mask] = dst_pts[i][best_mask]

    return selected_pts_gt, selected_dst, selected_src,masks