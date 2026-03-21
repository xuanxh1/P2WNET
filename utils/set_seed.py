"""
随机种子设置工具
用于确保实验的可重复性
"""
import random
import numpy as np
import torch
import torch.distributed as dist

def set_seed(seed):
    """
    设置所有随机数生成器的种子，确保实验可重复性
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 多GPU设置

    # 确保在使用 cuDNN 时产生确定性结果
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_seed_ddp(seed, is_distributed=False):
    """
    为分布式训练设置随机种子，每个进程使用不同的种子偏移
    
    参数:
        seed: 基础随机种子值
        is_distributed: 是否为分布式训练模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保在使用cuDNN时产生确定性结果
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 分布式训练中，增加额外的种子偏移以确保每个进程使用不同种子
    if is_distributed:
        seed_offset = dist.get_rank()
        random.seed(seed + seed_offset)
        np.random.seed(seed + seed_offset)
        torch.manual_seed(seed + seed_offset)