import json
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
# 设置CUDA线性代数库后端
torch.backends.cuda.preferred_linalg_library(backend='cusolver')
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
import random
from models.loss_k_pts import sim_matrix_loss_general
from utils.get_dataloder import get_dataloader
from utils.copy_all_files import copy_files_exclude
from utils.logger import setup_logger
from utils.cfg_management import merge_args_into_config
from utils.utils import get_obj_from_str
from train.utils_train.get_args import parse_args
from utils.print_model import print_one_model_summaries
from utils.loading import load_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from contextlib import nullcontext
from utils.set_seed import set_seed_ddp
from utils.get_lr_schdule import get_scheduler

def setup_distributed():
    """初始化分布式训练环境"""
    # 从环境变量获取分布式参数
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 判断是否启用分布式
    is_distributed = local_rank != -1
    
    if is_distributed:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(local_rank)
        # 初始化NCCL进程组
        dist.init_process_group(backend='nccl', init_method='env://')
        if rank == 0:
            print(f"分布式训练已初始化，world_size={world_size}")
    
    return is_distributed, local_rank, rank, world_size

def print_config(config, logger, indent=0):
    """递归打印配置信息"""
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(' ' * indent + f"{key}:")
            print_config(value, logger, indent + 2)
        else:
            logger.info(' ' * indent + f"{key}: {value}")

if __name__ == "__main__":
    # 初始化分布式环境
    is_distributed, local_rank, rank, world_size = setup_distributed()
    master_process = rank == 0  # 判断是否为主进程
    
    # 解析命令行参数
    args = parse_args()
    # 设置随机种子保证可复现
    seed = args.seed
    set_seed_ddp(seed, is_distributed)
    
    # 创建保存目录和日志器
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("P2WNET", args.save_dir, 0)
    logger1 = setup_logger("P2WNET-BEST-METRICS", args.save_dir, 0, filename='best_metrix_log')
    
    # 主进程备份训练代码
    if master_process:
        current_dir = str(Path(__file__).resolve().parent.parent)
        code_files_target_dir = os.path.join(args.save_dir,'code_files')
        copy_files_exclude(current_dir,code_files_target_dir)
    
    # 加载配置文件
    with open(args.config, "r") as f:
        config = json.load(f)
    merge_args_into_config(args, config)
    
    # 添加分布式配置信息
    config["distributed"] = {
        "enabled": is_distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "master_process": master_process
    }
    
    # 调整梯度累积步数适配分布式训练
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    if is_distributed and gradient_accumulation_steps > 1:
        # 确保步数可被进程数整除
        if gradient_accumulation_steps % world_size != 0:
            original_steps = gradient_accumulation_steps
            gradient_accumulation_steps = (original_steps // world_size) * world_size
            if master_process:
                logger.info(f"梯度累积步数已从 {original_steps} 调整为 {gradient_accumulation_steps}")
        # 每个进程分担一部分累积步数
        gradient_accumulation_steps //= world_size
        config["gradient_accumulation_steps"] = gradient_accumulation_steps
    
    # 主进程打印所有配置信息
    if master_process:
        logger.info('=================================================================================')
        for arg, value in vars(args).items():
            logger.info(f"  {arg}: {value}")
        logger.info('=================================================================================')
        print_config(config, logger)
        logger.info('=================================================================================')

    # 获取训练和验证数据加载器
    dataloader_result = get_dataloader(config, args)
    if len(dataloader_result) == 4:  # 分布式模式返回4个值
        training_loader, validation_loader, train_sampler, val_sampler = dataloader_result
    else:  # 单卡模式返回2个值
        training_loader, validation_loader = dataloader_result
        train_sampler, val_sampler = None, None

    # 设置训练设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 获取设备类型
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    # 初始化训练参数
    epoch = 0
    batch_size = args.batch_size
    num_epochs = config["epochs"]

    # 创建匹配模型
    model_obj = get_obj_from_str(config['model_name'])
    model = model_obj(
        num_features=config["num_features"],
        downsampling=config["downsample_factor"],
        imgs_color=config['imgs_color'],
        template_size=config['training_template_img_size'],
        search_size=config['training_search_img_size'],
        max_shape = config["max_shape"],
        d_model=config["d_model"],
        n_heads = config["n_heads"],
        layer_names= ['self', 'cross'] * config['attention_layers_num'],
        head_kernel_size = config['head_kernel_size'],
        num_head_layers = config['num_head_layers'],
        num_of_tasktoken = config['num_of_task_token'],
        num_of_predited_pts = config['num_of_predited_pts'],
        attention_drop_out_rate = config['attention_drop_out_rate'],
        cnn_act = config['cnn_activation'],
        att_act = config['att_activation'],
        cnn_norm = config['cnn_norm'],
        use_self_attention_before_predition=config['use_self_attention_before_predition'],
        cnn_conv2d_has_bias=config['cnn_conv2d_has_bias'],
        cnn_bn_has_bias=config['cnn_bn_has_bias'],
        att_scales=config['att_scales'],
        att_norm=config['att_norm'],
        att_conv2d_has_bias=config['att_conv2d_has_bias'],
        att_bn_has_bias=config['att_bn_has_bias'],
        attn_type=config['attn_type'],
        size_corr_kernel_size=config['size_corr_kernel_size'],
        use_share_encoder=config['use_share_encoder'],
        kernel_fn=config['kernel_fn'],
        num_features_predition_head = config['num_features_predition_head'],
    ).to(device)
    
    # 主进程打印模型结构和参数量
    if master_process:
        print_one_model_summaries(model, logger)
    
    # 用DDP包装模型实现分布式训练
    if is_distributed:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
        )
    
    # 创建AdamW优化器
    optimizer = AdamW(
        params=model.parameters(),
        lr=config["lr"],
        betas=(config["momentum"], 0.999),
        weight_decay=config["weight_decay"],
        eps=config["epsilon"],
    )

    # 使用统一的学习率调度器函数创建scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # 加载预训练权重或断点续训
    epoch = load_checkpoint(args, model, optimizer, scheduler, device)
    
    # 创建关键点预测损失函数
    loss_fn_obj = get_obj_from_str(config['loss_fn_name'])
    criterion = loss_fn_obj(
        size_sar=config['training_template_img_size'],
        size_optical=config['training_search_img_size'],
        stride=config["downsample_factor"],
        config_loss=config["loss"],
    ).to(device)
    
    # 创建相似度矩阵损失函数
    sim_matrix_criterion = sim_matrix_loss_general(
        size_sar=config['training_template_img_size'],
        size_optical=config['training_search_img_size'],
        stride=config["downsample_factor"],
        config_loss=config["loss"],
        scale_factor = config['scale_factor']
    ).to(device)
    
    # 创建关键点解析器
    box_parser_obj = get_obj_from_str(config['box_parser_name'])
    box_parser = box_parser_obj(stride=config["downsample_factor"]).to(device)

    # 主进程创建TensorBoard可视化
    writer = None
    if master_process:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(log_dir=args.save_dir)

    # 获取训练函数并开始训练
    train_n_epoches_obj = get_obj_from_str(config['train_k_epoches_fn_name'])
    
    train_n_epoches_obj(model,
                        training_loader,
                        validation_loader,
                        criterion,
                        sim_matrix_criterion,
                        optimizer,
                        scheduler,
                        box_parser,
                        config,
                        args,
                        logger,
                        logger1,
                        writer,
                        device,
                        num_epochs,
                        epoch,
                        is_distributed=is_distributed,
                        master_process=master_process,
                        train_sampler=train_sampler,
                        val_sampler=val_sampler)
    
    # 释放资源
    if writer and master_process:
        writer.close()
    
    if is_distributed:
        dist.destroy_process_group()


