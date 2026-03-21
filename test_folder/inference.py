
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import device as Device
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from models.box_parser import get_k_pts_BoxParser
from train.utils_train.get_args import parse_args
from utils.logger import setup_logger
from utils.utils import get_obj_from_str
from utils.cfg_management import merge_args_into_config
from utils.get_dataloder import get_val_dataloder
from train.utils_train.get_pts import get_4_pts_in_template_and_search_img
from train.utils_train.get_error import get_mace
PERCENTAGE_INTERVALS = [(0, 30), (30, 60), (60, 100)]
NUM_EVAL_REPEATS = 5


def evaluate_once(
    model,
    args,
    dataloader: DataLoader,
    box_parser: get_k_pts_BoxParser,
    size_template,
    dev: Device,
    k=16,
):
    """单次完整验证，返回每个样本的 MACE。"""
    model.eval()

    mace_list = []

    with torch.no_grad():
        for batch_data in tqdm(
            dataloader,
            total=len(dataloader),
            unit="batch",
            desc="[ 验证中 ]",
            colour="blue",
        ):
            imgs_search = batch_data[0]
            imgs_template = batch_data[1]
            tl_gt = batch_data[2]
            H_matrix = batch_data[3]


            imgs_search = imgs_search.to(dev, non_blocking=True)
            imgs_template = imgs_template.to(dev, non_blocking=True)
            tl_gt = tl_gt.to(dev, non_blocking=True)
            H_matrix = H_matrix.to(dev, non_blocking=True)

            pts_gt, _ = get_4_pts_in_template_and_search_img(tl_gt=tl_gt,
                                            size_template=size_template,
                                            H_matrix=H_matrix,
                                            inner_dis=config['inner_dis'],
                                            scale_factor=1)

            _, pred_score_map, pred_offset_map, x_search, x_template = model(
                imgs_search, imgs_template
            )
            pts_pred = box_parser(pred_score_map, pred_offset_map)

            mace_values = get_mace(pts_gt, pts_pred, k=k)
          
            mace_list.extend(mace_values.detach().cpu().tolist())
    return mace_list


def get_percentage_interval_means(values):
    """按 0-30%, 30-60%, 60-100% 计算区间均值。"""
    sorted_values = sorted(values)
    total_count = len(sorted_values)
    interval_means = {}

    for lower_percent, upper_percent in PERCENTAGE_INTERVALS:
        lower_index = int((lower_percent / 100) * total_count)
        upper_index = int((upper_percent / 100) * total_count)
        interval_values = sorted_values[lower_index:upper_index]
        key = f"{lower_percent}-{upper_percent}%"
        interval_means[key] = float(np.mean(interval_values)) if interval_values else 0.0

    return interval_means


def run_repeated_evaluation(
    model,
    args,
    dataloader: DataLoader,
    box_parser: get_k_pts_BoxParser,
    size_template,
    dev: Device,
    k=16,
    repeats=NUM_EVAL_REPEATS,
):
    """重复验证多次，返回聚合后的稳定统计值。"""
    mace_means = []
    interval_means_runs = {f"{l}-{u}%": [] for l, u in PERCENTAGE_INTERVALS}

    for repeat_idx in range(repeats):
        logger.info(f"开始第 {repeat_idx + 1}/{repeats} 次验证")
        mace_list = evaluate_once(
            model=model,
            args=args,
            dataloader=dataloader,
            box_parser=box_parser,
            size_template=size_template,
            dev=dev,
            k=k,
        )
        mace_mean = float(np.mean(mace_list)) if mace_list else 0.0
        mace_means.append(mace_mean)
        interval_means = get_percentage_interval_means(mace_list)
        for key in interval_means_runs:
            interval_means_runs[key].append(interval_means[key])

    final_interval_means = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in interval_means_runs.items()
    }
    final_mace_mean = float(np.mean(mace_means)) if mace_means else 0.0
    return final_mace_mean, final_interval_means


def save_simple_statistics(output_file_path, mace_mean, interval_means):
    """仅保存用户要求的统计结果。"""
    with open(output_file_path, "w") as f:
        f.write(f"mace_mean: {mace_mean:.6f}\n")
        f.write(f"interval_0_30_mean: {interval_means['0-30%']:.6f}\n")
        f.write(f"interval_30_60_mean: {interval_means['30-60%']:.6f}\n")
        f.write(f"interval_60_100_mean: {interval_means['60-100%']:.6f}\n")


def load_dataset(args, config):
    """加载验证数据集"""
    validation_loader, *_ = get_val_dataloder(config, args, split=args.split)
    return validation_loader


if __name__ == "__main__":
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("P2WNET_INFERENCE", args.save_dir, 0)
    
    # 加载配置文件
    with open(args.config, "r") as f:
        config = json.load(f)
    
    merge_args_into_config(args, config)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ).to(dev)
    model = nn.DataParallel(model)

    path_checkpoint = args.resume
    if path_checkpoint and path_checkpoint.exists():
        checkpoint = torch.load(path_checkpoint, map_location=dev)
        model_state = checkpoint["model_state"]
        current_state = model.state_dict()
        new_state_dict = {}

        for k, v in model_state.items():
            if k in current_state:
                if v.shape == current_state[k].shape:
                    new_state_dict[k] = v
                else:
                    logger.info(
                        f"跳过参数 {k}，形状不匹配: "
                        f"checkpoint 形状 {v.shape} vs 模型形状 {current_state[k].shape}"
                    )
            else:
                logger.info(f"跳过参数 {k}，在当前模型中未找到")

        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"从checkpoint加载了 {len(new_state_dict)}/{len(current_state)} 层")
    else:
        logger.info(f"Checkpoint 文件不存在: {path_checkpoint}")

    box_parser = get_k_pts_BoxParser(stride=config["downsample_factor"]).to(dev)
    args.batch_size = 1
    validation_loader = load_dataset(args, config)

    mace_mean, interval_means = run_repeated_evaluation(
        model=model,
        args=args,
        dataloader=validation_loader,
        box_parser=box_parser,
        size_template=config['training_template_img_size'],
        dev=dev,
        k=config['num_of_predited_pts'],
        repeats=NUM_EVAL_REPEATS,
    )

    output_file = os.path.join(args.save_dir, "simple_eval_statistics.txt")
    save_simple_statistics(output_file, mace_mean, interval_means)
    logger.info(f"结果已保存: {output_file}")
