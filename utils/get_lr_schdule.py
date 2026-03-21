"""
学习率调度器配置工具
支持warmup+cosine、step和恒定学习率等多种调度策略
"""
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, StepLR

def get_scheduler(optimizer, config):
    """
    根据配置创建学习率调度器
    
    参数:
        optimizer: PyTorch优化器
        config: 配置字典，包含学习率调度相关参数
    
    返回:
        scheduler: 学习率调度器对象
    
    支持的调度类型:
        - warmup_cosine: 预热+余弦退火
        - step: 阶梯式衰减
        - none: 恒定学习率
    
    配置参数说明:
        - lr_schedule_type: 调度器类型
        - epochs: 总训练轮数
        - warmup_epochs: 预热轮数(warmup_cosine类型需要)
        - warmup_start_factor: 预热起始因子(warmup_cosine类型需要)
        - eta_min: 最小学习率(warmup_cosine类型需要)
        - lr_decrease_period: 学习率衰减周期(step类型需要)
        - lr_gamma: 学习率衰减系数(step类型需要)
    """
    lr_schedule_type = config.get('lr_schedule_type', 'step')
 
    if lr_schedule_type == 'warmup_cosine':
        warmup_epochs = config['warmup_epochs']
        cosine_epochs = config['epochs'] - warmup_epochs
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=config['warmup_start_factor'],
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=config['eta_min']
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
    elif lr_schedule_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config['lr_decrease_period'],
            gamma=config.get('lr_gamma', 0.7)
        )
        
    elif lr_schedule_type == 'none':
        # 返回一个恒定学习率的调度器（LambdaLR with lambda=1.0）
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0
        )
        
    else:
        raise ValueError(f"未知的 lr_schedule_type: {lr_schedule_type}")
    
    return scheduler
