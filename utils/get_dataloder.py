"""
数据加载器配置工具
根据配置创建训练和验证数据加载器，支持分布式训练
"""
from re import S
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.augmentation_utils import get_train_transform_fn, get_val_transform_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.HomographyEstimationDataset import *
from dataset.dataset_utils import *
from torch.utils.data import DataLoader

def get_val_dataloder(config,args,split='test'):
    """
    创建验证数据加载器
    
    参数:
        config: 配置字典，包含数据集路径、图像尺寸等参数
        args: 命令行参数，包含batch_size和num_workers
        split: 数据集划分，'val'或'test'
    
    返回:
        validation_loader: 验证数据加载器
        val_sampler: 分布式采样器(如果使用分布式训练)，否则为None
    """
    val_transform_fn  = get_val_transform_fn(config)
    
    dataset_val = Homography_Dataset_runtime(
        root_list = config["dataset_augmentations"]["dataset_list"],
        split=split,
        search_size = config["training_search_img_size"],
        template_patch_size = config["training_template_img_size"],
        transform = val_transform_fn,
        x_flip = config['x_flip'],
        y_flip = config['y_flip'],
        color  = config['imgs_color'],
        uni_modality=config["dataset_augmentations"]['uni_modality'],
        min_overlap_ratio=1,
    )
    
    
    is_distributed = config.get("distributed", {}).get("enabled", False)
    # 验证集也使用分布式采样器(但不shuffle)
    if is_distributed:
        val_sampler = DistributedSampler(dataset_val, shuffle=False)
    else:
        val_sampler = None

    validation_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=min(os.cpu_count(), args.num_workers),
        pin_memory=True,
        collate_fn=None,
        drop_last=True,
    )

    # 返回数据加载器和采样器
    if is_distributed:
        return validation_loader, val_sampler
    else:
        return validation_loader, None

def get_train_dataloder(config,args,split='train'):
    """
    创建训练数据加载器
    
    参数:
        config: 配置字典，包含数据集路径、图像尺寸、数据增强参数等
        args: 命令行参数，包含batch_size和num_workers
        split: 数据集划分，默认为'val'
    
    返回:
        training_loader: 训练数据加载器
        train_sampler: 分布式采样器(如果使用分布式训练)，否则为None
    """
    train_transform_fn  = get_train_transform_fn(config)

    dataset_train = Homography_Dataset_runtime(
        root_list = config["dataset_augmentations"]["dataset_list"],
        split=split,
        search_size = config["training_search_img_size"],
        template_patch_size = config["training_template_img_size"],
        transform = train_transform_fn,
        x_flip = config['x_flip'],
        y_flip = config['y_flip'],
        color  = config['imgs_color'],
        uni_modality=config["dataset_augmentations"]['uni_modality'],
        min_overlap_ratio=1,
    )
        
    # 配置分布式采样器
    is_distributed = config.get("distributed", {}).get("enabled", False)
    if is_distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        shuffle = False  # 使用sampler时shuffle参数必须为False
    else:
        train_sampler = None
        shuffle = True

    training_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=min(os.cpu_count(), args.num_workers),
        pin_memory=True,
        collate_fn=None,
        drop_last=True
    )

    # 返回数据加载器和采样器
    if is_distributed:
        return training_loader, train_sampler
    else:
        return training_loader, None

def get_dataloader(config,args):
    """
    同时创建训练和验证数据加载器
    
    参数:
        config: 配置字典
        args: 命令行参数
        split: 数据集划分
    
    返回:
        training_loader: 训练数据加载器
        validation_loader: 验证数据加载器
        train_sampler: 训练分布式采样器(如果使用分布式训练)
        val_sampler: 验证分布式采样器(如果使用分布式训练)
    """
    is_distributed = config.get("distributed", {}).get("enabled", False)

    training_loader,train_sampler = get_train_dataloder(config,args,split='train')
    validation_loader,val_sampler = get_val_dataloder(config,args,split='val')

    # 返回数据加载器和采样器
    if is_distributed:
        return training_loader, validation_loader, train_sampler, val_sampler
    else:
        return training_loader, validation_loader, None, None
