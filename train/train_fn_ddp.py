import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import  device as Device
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.box_parser import get_k_pts_BoxParser
import torch.nn as nn
from utils.utils import get_obj_from_str,write_epoch_results
torch.set_printoptions(precision=2)
from utils.save_best_model import save_best_checkpoint
from train.utils_train.get_pts import *
from train.utils_train.get_grid_like_points import *
from train.utils_train.get_error import *
from utils.logger import save_metrics_periodically
import time
import torch.distributed as dist

def train_one_epoch(
    model,
    args,
    dataloader: DataLoader,
    criterion,
    sim_matrix_criterion,
    optimizer,
    box_parser: get_k_pts_BoxParser,
    size_sar,
    k,
    inner_dis,
    scale_factor,
    dev: Device,
    max_grad_value=10,
) :
    model.train()
    
    # 统一管理所有统计量
    running_metrics = {
        "train_loss_sim": 0.0,
        "train_loss_cls": 0.0,
        "train_loss_offset": 0.0,
        "train_loss_total": 0.0,
        "train_error_l2_overall": 0.0,
        "train_error_l2_in": 0.0, 
        "train_error_l2_out": 0.0,
        "train_mace": 0.0 ,
        "train_mace_by_dlt": 0.0,
        "train_mace_by_usanc": 0.0,
    }

    # 样本计数器（总体和按数据集）
    sample_counts = {"total": 0}
    
    train_error_l2_overall = []
    train_error_l2_in = []
    train_error_l2_out = []

    idx = 0
    
    for imgs_search, imgs_template, tl_gt, H_matrix,*_ in tqdm(
        dataloader,
        total=len(dataloader),
        unit="batch",
        desc="[ train ]",
        colour="blue",
        mininterval=20,
    ):
        batch_size = tl_gt.shape[0]
        idx += batch_size
        sample_counts["total"] += batch_size
        
        # Transfer data to target device
        imgs_search = imgs_search.to(dev, non_blocking=True)
        imgs_template = imgs_template.to(dev, non_blocking=True)
        tl_gt = tl_gt.to(dev, non_blocking=True)
        H_matrix = H_matrix.to(dev, non_blocking=True)
        with torch.no_grad():
            pts_gt,pts_org = get_4_pts_in_template_and_search_img(tl_gt,size_sar,H_matrix,inner_dis,scale_factor=scale_factor)
            pts_gt = pts_gt.to(dev, non_blocking=True)
            pts_org = pts_org.to(dev, non_blocking=True)

        optimizer.zero_grad()
        
       
        pred_sim_matrix ,pred_score_map, pred_offset_map,x_search,x_template = model(imgs_search, imgs_template)
        
        loss_cls, loss_offset, loss_total = criterion(
            (pred_score_map, pred_offset_map),
            pts_gt,
        )
        sim_loss = sim_matrix_criterion(pred_sim_matrix, tl_gt, H_matrix)
        loss_total = loss_total + sim_loss
        
        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_value, norm_type=2.0)
        optimizer.step()

        with torch.no_grad():
            pts_pred = box_parser(pred_score_map, pred_offset_map)
            l2_error, l2_error_in, l2_error_out = get_l2_error_in_and_out(
                pts_gt=pts_gt,
                pts_pred=pts_pred,
                search_size=imgs_search.shape[-2:],
            )
            MACE = get_mace(pts_gt,pts_pred,k)
          
            running_metrics["train_loss_sim"] += sim_loss.item() * batch_size
            running_metrics["train_loss_cls"] += loss_cls.item() * batch_size
            running_metrics["train_loss_offset"] += loss_offset.item() * batch_size
            running_metrics["train_loss_total"] += loss_total.item() * batch_size
            train_error_l2_overall.extend(l2_error.tolist())
            train_error_l2_in.extend(l2_error_in.tolist())
            train_error_l2_out.extend(l2_error_out.tolist())
            running_metrics['train_mace'] += MACE.mean().item() * batch_size
            
    total_samples = sample_counts["total"]
    avg_metrics = {key: value / total_samples for key, value in running_metrics.items()}

    error_l2_overall = sum(train_error_l2_overall) / len(train_error_l2_overall) if len(train_error_l2_overall) > 0 else 0.0
    error_l2_in = sum(train_error_l2_in) / len(train_error_l2_in) if len(train_error_l2_in) > 0 else 0.0
    error_l2_out = sum(train_error_l2_out) / len(train_error_l2_out) if len(train_error_l2_out) > 0 else 0.0
    avg_metrics["train_error_l2_overall"] = error_l2_overall
    avg_metrics["train_error_l2_in"] = error_l2_in
    avg_metrics["train_error_l2_out"] = error_l2_out


    # 合并结果：全局指标和按数据集分类的指标
    result = {
        "global": avg_metrics,
        "sample_counts": sample_counts
    }

    return result

def vaildation_one_epoch(
    model,
    args,
    dataloader: DataLoader,
    box_parser: get_k_pts_BoxParser,
    size_sar,
    k,
    inner_dis,
    scale_factor,
    logger,
    dev: Device
):
    """验证一轮，预测k个关键点"""
    model.eval()  # 切换到评估模式
    
    # 初始化混合精度
    if args.use_fp16:
        from torch.cuda.amp import autocast
    
    # 获取分布式训练信息
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    
    # 存储多轮验证结果
    all_rounds_results = []
    all_rounds_mace_by_usanc_lists = []
    
    for round_idx in range(1):  # 验证1轮
        # 重置指标累加器
        running_metrics = {
            "val_error_l2_overall": 0.0,
            "val_error_l2_in": 0.0,
            "val_error_l2_out": 0.0,
            "val_mace": 0.0,
            "val_mace_by_dlt": 0.0,
            "val_mace_by_usanc": 0.0
        }

        # 重置误差列表
        mace_by_usanc_list = []
        val_error_l2_overall = []
        val_error_l2_in = []
        val_error_l2_out = []

        # 重置样本计数
        sample_counts = {"total": 0}

        # 禁用梯度计算
        with torch.no_grad():
            # 遍历验证数据
            for imgs_search, imgs_template, tl_gt, H_matrix,*_  in tqdm(
                dataloader,
                total=len(dataloader),
                unit="batch",
                desc=f"[Rank {rank}] val {round_idx+1}/1",
                colour="blue",
                mininterval=20,
                disable=(rank != 0),  # 非主进程不显示进度条
            ):
                batch_size = tl_gt.shape[0]
                sample_counts["total"] += batch_size
                
                # 数据异步传输到GPU
                imgs_search = imgs_search.to(dev, non_blocking=True)
                imgs_template = imgs_template.to(dev, non_blocking=True)
                tl_gt = tl_gt.to(dev, non_blocking=True)
                H_matrix = H_matrix.to(dev, non_blocking=True)
                pts_gt, pts_org = get_4_pts_in_template_and_search_img(
                    tl_gt, size_sar, H_matrix, inner_dis, scale_factor=scale_factor
                )
            
                pts_gt = pts_gt.to(dev, non_blocking=True)
                pts_org = pts_org.to(dev, non_blocking=True)

                # 前向传播
                if args.use_fp16:
                    with autocast():  # 混合精度推理
                        _, pred_score_map, pred_offset_map, x_search, x_template = model(
                            imgs_search, imgs_template
                        )
                else:
                    _, pred_score_map, pred_offset_map, x_search, x_template = model(
                        imgs_search, imgs_template
                    )
                
                # 解析预测的关键点
                pts_pred = box_parser(pred_score_map, pred_offset_map)

                # 计算评估指标
                l2_error, l2_error_in, l2_error_out = get_l2_error_in_and_out(
                    pts_gt=pts_gt,
                    pts_pred=pts_pred,
                    search_size=imgs_search.shape[-2:],
                )
                
                mace = get_mace(pts_gt, pts_pred, k)
                

                # 累加当前batch的指标
                val_error_l2_overall.extend(l2_error.tolist())
                val_error_l2_in.extend(l2_error_in.tolist())
                val_error_l2_out.extend(l2_error_out.tolist())
                running_metrics["val_mace"] += mace.mean().item() * batch_size
               
           
        # 分布式训练同步各进程数据
        if is_distributed:
            dist.barrier()  # 等待所有进程完成
            
            # 聚合各进程的列表数据
            def gather_list_tensor(local_list):
                # 先收集所有进程的列表长度
                local_size = torch.tensor([len(local_list)], dtype=torch.long, device=dev)
                size_list = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(world_size)]
                dist.all_gather(size_list, local_size)
                size_list = [int(s.item()) for s in size_list]
                max_size = max(size_list) if size_list else 0
                
                # 如果所有进程都没有数据，直接返回空列表
                if max_size == 0:
                    return []
                
                # 创建或填充tensor到max_size
                if len(local_list) == 0:
                    local_tensor = torch.zeros(max_size, dtype=torch.float32, device=dev)
                elif len(local_list) < max_size:
                    local_tensor = torch.tensor(local_list, dtype=torch.float32, device=dev)
                    padding = torch.zeros(max_size - len(local_list), dtype=torch.float32, device=dev)
                    local_tensor = torch.cat([local_tensor, padding])
                else:
                    local_tensor = torch.tensor(local_list, dtype=torch.float32, device=dev)
                
                # 收集所有进程的数据
                gathered = [torch.zeros(max_size, dtype=torch.float32, device=dev) for _ in range(world_size)]
                dist.all_gather(gathered, local_tensor)
                
                # 合并并去除填充
                merged = []
                for tensor, size in zip(gathered, size_list):
                    if size > 0:
                        merged.extend(tensor[:size].cpu().tolist())
                return merged
            
            # 同步样本总数
            count_tensor = torch.tensor(sample_counts["total"], dtype=torch.long, device=dev)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            sample_counts["total"] = count_tensor.item()
            
            # 同步累加指标
            for key in sorted(running_metrics.keys()):
                metric_tensor = torch.tensor(running_metrics[key], dtype=torch.float32, device=dev)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                running_metrics[key] = metric_tensor.item()
            
            
            # 同步误差列表
            mace_by_usanc_list = gather_list_tensor(mace_by_usanc_list)
            val_error_l2_overall = gather_list_tensor(val_error_l2_overall)
            val_error_l2_in = gather_list_tensor(val_error_l2_in)
            val_error_l2_out = gather_list_tensor(val_error_l2_out)
            
            dist.barrier()  # 再次同步
           


        # 计算当前轮次的平均指标
        total_samples = sample_counts["total"]
        if total_samples == 0:
            if rank == 0:
                logger.warning("警告：总样本数为0！")
            total_samples = 1
        
        round_avg_metrics = {key: value / total_samples for key, value in running_metrics.items()}
      
        # 计算L2误差的平均值
        error_l2_overall = sum(val_error_l2_overall) / len(val_error_l2_overall) if len(val_error_l2_overall) > 0 else 0.0
        error_l2_in = sum(val_error_l2_in) / len(val_error_l2_in) if len(val_error_l2_in) > 0 else 0.0
        error_l2_out = sum(val_error_l2_out) / len(val_error_l2_out) if len(val_error_l2_out) > 0 else 0.0
        
        round_avg_metrics["val_error_l2_overall"] = error_l2_overall
        round_avg_metrics["val_error_l2_in"] = error_l2_in
        round_avg_metrics["val_error_l2_out"] = error_l2_out

        # 保存当前轮次的结果
        round_result = {
            "metrics": round_avg_metrics,
            "sample_counts": sample_counts,
            "mace_by_usanc_list": mace_by_usanc_list,
            "val_error_l2_overall_list": val_error_l2_overall,
        }
        
        all_rounds_results.append(round_result)
        all_rounds_mace_by_usanc_lists.extend(mace_by_usanc_list)
        
        # 主进程记录当前轮次结果
        if rank == 0:
            logger.info(f"Round {round_idx+1}/1 completed:")
            logger.info(f"  L2 Overall: {error_l2_overall:.4f}")
            logger.info(f"  L2 In: {error_l2_in:.4f}")
            logger.info(f"  L2 Out: {error_l2_out:.4f}")
            logger.info(f"  MACE: {round_avg_metrics['val_mace']:.4f}")

    # 计算所有轮次的平均指标
    final_avg_metrics = {}
    for key in all_rounds_results[0]["metrics"].keys():
        values = [round_result["metrics"][key] for round_result in all_rounds_results]
        final_avg_metrics[key] = sum(values) / len(values)

    # 合并所有轮次的误差列表
    all_val_error_l2_overall = []
    for round_result in all_rounds_results:
        all_val_error_l2_overall.extend(round_result["val_error_l2_overall_list"])
    
    # 主进程输出统计分析
    if rank == 0:

        logger.info("=== 最终平均结果 (1轮) ===")
        logger.info(f"平均L2 Overall: {final_avg_metrics['val_error_l2_overall']:.4f}")
        logger.info(f"平均L2 In: {final_avg_metrics['val_error_l2_in']:.4f}")
        logger.info(f"平均L2 Out: {final_avg_metrics['val_error_l2_out']:.4f}")
        logger.info(f"平均MACE: {final_avg_metrics['val_mace']:.4f}")
    
    # 统计总样本数
    total_sample_counts = {"total": sum(round_result["sample_counts"]["total"] for round_result in all_rounds_results)}
    

    # 返回验证结果
    result = {
        "global": final_avg_metrics,
        "sample_counts": total_sample_counts,
        "mace_by_usanc_list": all_rounds_mace_by_usanc_lists,
        "all_rounds_details": all_rounds_results,
    }

    return result

def train_n_epoches(
    model,
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
    dev,
    num_epochs,
    epoch=0,
    is_distributed=False,
    master_process=True,
    train_sampler=None,
    val_sampler=None,
):
    # 初始化最佳指标记录
    best_loss = 10000
    best_mace = 10000
    best_mace_by_dlt = 10000
    best_mace_by_usanc = 10000
    
    # 动态加载训练和验证函数
    train_fn_obj = get_obj_from_str(config['train_fn_name'])
    validation_fn_obj = get_obj_from_str(config['val_fn_name'])

    while epoch < num_epochs:  # 开始训练循环
        
        # 设置分布式采样器的epoch
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_distributed and val_sampler is not None:
            val_sampler.set_epoch(epoch)

        # 记录训练开始时间
        train_start_time = time.time()
        
        #训练一个epoch
        train_results = train_fn_obj(
            args=args,
            model=model,
            dataloader=training_loader,
            criterion=criterion,
            sim_matrix_criterion=sim_matrix_criterion,
            optimizer=optimizer,
            box_parser=box_parser,
            size_sar=config['training_template_img_size'],
            k=config['num_of_predited_pts'],
            dev=dev,
            inner_dis=config['inner_dis'],
            scale_factor=config['scale_factor'],
        )
        
        # 记录训练耗时
        train_time = time.time() - train_start_time
        if master_process:
            logger.info(f"训练一个epoch耗时: {train_time:.2f}秒")
        
        # 检查梯度状态
        if master_process and config.get('check_gradients', False):
            no_grad_params = no_grad_parameters(model)
            if no_grad_params:
                logger.info(f"模型中没有梯度的参数: {len(no_grad_params)}")
                for name in no_grad_params[:10]:
                    logger.info(f"  - {name}")
                if len(no_grad_params) > 10:
                    logger.info(f"  ... 还有 {len(no_grad_params) - 10} 个参数")
        
        # Extract global metrics from train results
        avg_metrics = train_results["global"]
        
        # Log global metrics - 只在主进程中记录日志
        if master_process:
            for key, value in sorted(avg_metrics.items()):
                formatted_value = f"{value:>12.7f}"
                formatted_key = f"{key:<30}"
                logger.info(f"Epoch: {epoch:>3d} [Train] {formatted_key} {formatted_value}")
                
                if writer:
                    writer.add_scalars(
                        f"{key}",
                        {f"{key}": value},
                        epoch,
                    )
                    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint - 只在主进程中保存检查点
        if master_process and args.save_period > 0 and (epoch + 1) % args.save_period == 0:
            dir_save: Path = args.save_dir
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }

            checkpoint_filename = Path(f"{epoch + 1:03d}.pt")
            logger.info(
                f"[{epoch + 1:3d}/{num_epochs:3d}] Saving checkpoint to {checkpoint_filename}."
            )
            torch.save(checkpoint, dir_save / checkpoint_filename)

          
        if (epoch + 1) % args.val_period == 0 and args.is_val:
            # 记录验证开始时间
            val_start_time = time.time()
            
            val_results = validation_fn_obj(
                model=model,
                args=args,
                dataloader=validation_loader,
                box_parser=box_parser,
                size_sar=config['training_template_img_size'],
                k=config['num_of_predited_pts'],
                inner_dis=config['inner_dis'],
                scale_factor=config['scale_factor'],
                dev=dev,
                logger=logger,
            )
           
            
            # 记录验证结束时间
            val_time = time.time() - val_start_time
            if master_process:
                logger.info(f"验证一个epoch耗时: {val_time:.2f}秒")
            
            # Extract global metrics from validation results
            avg_metrics_val = val_results["global"]

            error_l2_val = avg_metrics_val['val_error_l2_overall']
            mace_by_usanc = avg_metrics_val['val_mace_by_usanc']
            mace = avg_metrics_val['val_mace']
            mace_by_dlt = avg_metrics_val['val_mace_by_dlt']
            
            # 只在主进程中执行日志记录和检查点保存
            if master_process:
                # Log global validation metrics
                for key, value in sorted(avg_metrics_val.items()):
                    formatted_value = f"{value:>12.7f}"
                    formatted_key = f"{key:<30}"
                    logger.info(f"Epoch: {epoch:>3d} [Val] {formatted_key} {formatted_value}")
                    
                    if writer:
                        writer.add_scalars(
                            f"{key}",
                            {f"{key}": value},
                            epoch,
                        )
                
                # 只有当找到更好的结果时才执行统计分析打印，使用logger1
                if mace_by_usanc < best_mace_by_usanc:
                    # 添加全局统计分析打印
                    logger1.info('*******************************************************************************************************************************************')
                
                epoch_data = {
                    'epoch': epoch + 1,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                # Add global validation metrics to epoch data
                epoch_data.update(avg_metrics_val)

                txt_save_path = os.path.join(args.save_dir, 'training_data.txt')
                
                write_epoch_results(txt_save_path, epoch_data, header=(epoch == 0))
                
                best_loss = save_best_checkpoint(
                    current_loss=error_l2_val,
                    best_loss=best_loss,
                    epoch=epoch,
                    num_epochs=num_epochs,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_dir_checkpoint=args.save_dir / 'best.pt',
                    save_dir_only_model=args.save_dir / 'best_only_model.pt',
                    logger=logger,
                )

                best_mace_by_usanc = save_best_checkpoint(
                    current_loss=mace_by_usanc,
                    best_loss=best_mace_by_usanc,
                    epoch=epoch,
                    num_epochs=num_epochs,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_dir_checkpoint=args.save_dir / 'best_most_marginal.pt',
                    save_dir_only_model=args.save_dir / 'best_most_marginal_only_model.pt',
                    logger=logger,
                )
                best_mace = min(best_mace, mace)
                best_mace_by_dlt = min(best_mace_by_dlt, mace_by_dlt)
                logger.info(f"min_error_l2 {best_loss}")
                logger.info(f"best_mace {best_mace}")
                logger.info(f"best_mace_by_dlt {best_mace_by_dlt}")
                logger.info(f"best_mace_by_usanc {best_mace_by_usanc}")
                metrics_dict = {"best_loss":best_loss,"best_mace":best_mace,"best_mace_by_dlt":best_mace_by_dlt,"best_mace_by_usanc":best_mace_by_usanc}

                save_metrics_periodically(
                    metrics_dict,
                    dir_text=os.path.join(args.save_dir),
                    epoch=epoch,
                    period=25,
                    logger=logger
                )
            else:
                best_mace = min(best_mace, mace)
                best_mace_by_dlt = min(best_mace_by_dlt, mace_by_dlt)
                best_mace_by_usanc = min(best_mace_by_usanc, mace_by_usanc)
                best_loss = min(best_loss, error_l2_val)
            
        scheduler.step()
  
        epoch += 1
       
