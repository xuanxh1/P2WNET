"""
模型检查点加载工具
支持完整模型加载、仅编码器加载、参数匹配等功能
"""
import torch
from pathlib import Path

def load_checkpoint(args, model, optimizer, scheduler, dev):
    """
    只加载第一阶段模型的检查点。
    Args:
        args: 包含resume路径和加载选项的参数对象
        model: 第一阶段模型
        optimizer: 第一阶段优化器
        scheduler: 第一阶段调度器
        dev: 用于映射加载检查点的设备
    
    Returns:
        epoch: 加载的轮次（如果适用），否则返回None
    """
    epoch = 0

    # 第一阶段加载
    path_checkpoint = Path(args.resume) if args.resume else None
    if path_checkpoint and path_checkpoint.exists():
        print(f"[0/3] 在 {path_checkpoint} 找到检查点")
        try:
            checkpoint = torch.load(path_checkpoint, map_location=dev)
        except Exception as e:
            print(f"[错误] 加载检查点失败: {e}")
            return None

        if "model_state" not in checkpoint:
            print(f"[错误] 检查点缺少 'model_state' 键")
            return None

       
        print("[信息] 使用参数匹配加载完整模型")
        model_state = checkpoint["model_state"]
        current_model_state = model.state_dict()
        loaded_params, skipped_params = [], []

        # 加载所有匹配的参数
        for key in current_model_state.keys():
            # 尝试多种可能的checkpoint key匹配方式
            possible_keys = [
                key,  # 直接匹配
                key[7:] if key.startswith('module.') else f"module.{key}",  # module前缀转换
            ]
            
            # 找到匹配的checkpoint key
            source_key = None
            source_param = None
            for candidate_key in possible_keys:
                if candidate_key in model_state:
                    source_key = candidate_key
                    source_param = model_state[candidate_key]
                    break
            
            if source_param is not None:
                # 检查参数尺寸是否匹配
                current_param = current_model_state[key]
                if current_param.shape == source_param.shape:
                    current_model_state[key] = source_param
                    loaded_params.append(f"{source_key} -> {key}")
                else:
                    skipped_params.append(f"{key} (尺寸不匹配: 检查点 {source_param.shape} vs 模型 {current_param.shape})")
            else:
                skipped_params.append(f"{key} (在检查点中未找到)")

        model.load_state_dict(current_model_state, strict=False)
        print(f"[信息] 成功加载 {len(loaded_params)} 个参数:")
        for param in loaded_params[:5]:
            print(f"       ✓ {param}")
        if len(loaded_params) > 5:
            print(f"       ... 以及 {len(loaded_params) - 5} 个更多参数")

        if skipped_params:
            print(f"[信息] 跳过的参数 ({len(skipped_params)} 个参数):")
            for param in skipped_params[:5]:
                print(f"       ○ {param}")
            if len(skipped_params) > 5:
                print(f"       ... 以及 {len(skipped_params) - 5} 个更多参数")
            print(f"[信息] 这些参数将保持其初始化值")
        else:
            print(f"[信息] 所有参数都已从检查点中找到并加载")
        
        print(f"[1/3] 成功使用参数匹配加载完整模型")

        if not args.not_load_epoch and "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            print(f"[2/3] 从第 {epoch} 轮加载检查点")
            if "optimizer_state" in checkpoint and "scheduler_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if "_schedulers" in checkpoint["scheduler_state"] or not hasattr(scheduler, '_schedulers'):
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                    print(f"[3/3] 成功加载模型、优化器和调度器状态")
                else:
                    print(f"[3/3] 成功加载模型和优化器状态（调度器版本不兼容，将使用新的调度器）")
            else:
                print(f"[警告] 检查点中缺少优化器或调度器状态")
        else:
            print(f"[3/3] 成功加载模型（跳过优化器和调度器）")
    else:
        print("未找到第一阶段模型检查点")
    
    return epoch