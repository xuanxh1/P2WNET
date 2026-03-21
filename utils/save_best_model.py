"""
最佳模型保存工具
在验证损失降低时自动保存最佳检查点
"""
import torch
from pathlib import Path
from typing import Dict, Any

def save_best_checkpoint(
    current_loss: float,
    best_loss: float,
    epoch: int,
    num_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    save_dir_checkpoint,
    save_dir_only_model,
    logger: Any
) -> float:
    """
    当前损失低于最佳损失时保存检查点
    
    参数:
        current_loss: 当前验证损失
        best_loss: 目前为止的最佳损失
        epoch: 当前轮次
        num_epochs: 总轮次
        model: 待保存的模型
        optimizer: 待保存的优化器
        scheduler: 待保存的学习率调度器
        save_dir_checkpoint: 完整检查点保存路径
        save_dir_only_model: 仅模型保存路径
        logger: 日志记录器
    
    返回:
        更新后的最佳损失值
    """
    if current_loss < best_loss:
        best_loss = current_loss
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }

        logger.info(
            f"[{epoch + 1:3d}/{num_epochs:3d}] Saving best checkpoint to best.pt."
        )
        torch.save(checkpoint, save_dir_checkpoint)
        torch.save(model, save_dir_only_model)

    return best_loss