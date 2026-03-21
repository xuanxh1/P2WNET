import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
import torch
from torch import  device as Device
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.box_parser import get_k_pts_BoxParser
from torchvision import transforms
import torch.nn as nn
from torchvision import utils
from utils.draw import draw_lines
from train.utils_train.get_pts import get_4_pts_in_template_and_search_img
#设置打印时， torch变量最多显示两位小数
from train.utils_train.get_error import get_l2_error_in_and_out

torch.set_printoptions(precision=2)
transform = transforms.Compose([
    transforms.ToTensor(),              # 将PIL Image或numpy.ndarray转换为tensor，并归一化至[0,1]
    transforms.Normalize((0.5,), (0.5,)), # 使得像素值分布在[-1,1]之间
])
import time
import random
import numpy as np
import torch
import numpy as np
from utils.overlay_image_pairs import overlay_image_pairs
import kornia
from utils.utils import get_obj_from_str
from train.utils_train.get_args import parse_args
from utils.get_dataloder import get_val_dataloder
from utils.cfg_management import merge_args_into_config
from utils.logger import setup_logger
import torch.nn.functional as F
from utils.set_seed import set_seed

def restore_img(image, H, device):
    """
    将一张经过单应性变换过的图像复原， H 为变换所用的矩阵
    Args:
        image Tensor [bs,1,h,w]
        H Tensor [bs,3,3]
    Returns:
        Tensor [bs,1,h,w] 反单应性变换后的图像
    """

    b, c, h, w = image.shape
    # Create normalized meshgrid
    meshgrid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid_x, grid_y = meshgrid[0].to(device, non_blocking=True), meshgrid[1].to(device, non_blocking=True)
    ones = torch.ones_like(grid_x).to(device, non_blocking=True)
    grid = torch.stack([grid_y,grid_x, ones], dim=-1).unsqueeze(dim=0)
    
    grid = grid.repeat(b, 1, 1, 1).to(torch.float32)
    
    # Transform the grid
    transformed_grid = torch.matmul(grid.view(b, -1, 3), H.transpose(-1, -2))
    divisor = transformed_grid[:, :, 2].unsqueeze(dim=-1)
    transformed_grid = transformed_grid[:, :, :2] / divisor
    transformed_grid = transformed_grid.view(b, h, w, 2)
    # transformed_grid = torch.floor(transformed_grid[...,[1, 0 ]])
 
    # normalize and reshape
    transformed_grid[..., 0] = (transformed_grid[..., 0] / (w - 1)) * 2 - 1
    transformed_grid[..., 1] = (transformed_grid[..., 1] / (h - 1)) * 2 - 1 
      
    # Sample grid
    transformed_image = F.grid_sample(image, transformed_grid,align_corners=False,mode='nearest')
    
    return transformed_image

def get_gt_mask(template_img_size, search_img_size, gt_tl,H_matrix, scale_factor_4_sub_img, stride,device):
    """
    Args:
        template_img_size :(h,w) 模板图的大小
        search_img_size :(H,W) 原图的大小
        gt_tl Tensor :[b,2] 记录着若干组左上角真值
        H_matrix Tensor :[b,3,3] 单应性变换矩阵
        scale_factor_4_sub_img :[1] 对sub img 的缩放倍数
        stride :[1] 模型对输入图像的放缩倍数
        device   
    Returns:
        Tensor [b,1,H,W] 一张真值掩码
    """
    bs = gt_tl.shape[0]
    h_o, w_o = search_img_size
    h_sub, w_sub = template_img_size

    h_sub = h_sub //scale_factor_4_sub_img
    w_sub = w_sub //scale_factor_4_sub_img

    mask = torch.zeros(size=(bs,1,h_o*3,w_o*3)).to(device, non_blocking=True)
    ones = torch.ones(size=(bs,1,h_sub,w_sub)).to(device, non_blocking=True)

    for i in range(bs):
        mask[i,0,int(gt_tl[i,0]):int(gt_tl[i,0])+h_sub, int(gt_tl[i,1]):int(gt_tl[i,1])+w_sub] = ones[i,:,:,:]
        
    mask = restore_img(mask,H_matrix,device)
    #只返回与larger图片同样大小的gt_mask
    mask = mask[:,:,:h_o,:w_o]
    mask = F.interpolate(mask, scale_factor = 1/stride , mode='nearest')

    return mask

def get_most_marginal_conner_pts(H_pred,template_img_size):
    """
    Args:
    pts_org (torch.Tensor): Original points, shape [bs, 4, 2] (h, w order)
    pts_pred (torch.Tensor): Predicted points, shape [bs, 4, 2] (h, w order)
    H_gt (torch.Tensor): Ground truth homography matrix, shape [bs, 3, 3]
    template_img_size (tuple): Template image size (h, w)

    Returns:
    torch.Tensor: Mean L2 error for each batch, shape [bs]
    """
    h,w = template_img_size
    b = H_pred.shape[0]
    most_marginal_corners = torch.tensor([[0, 0],
                                        [w-1, 0],
                                        [0, h-1],
                                        [w-1, h-1]], dtype=H_pred.dtype, device=H_pred.device).unsqueeze(dim=0).repeat(b,1,1)
    
    most_marginal_corners_back_pred = kornia.geometry.transform_points(torch.inverse(H_pred),most_marginal_corners)
    
    return most_marginal_corners_back_pred[:,:,[1,0]]

def save_img_func(imgs_search, imgs_template, pts_predited, pts_gt, most_marginal_4_pts_gt, 
                 search_img_size, template_img_size, tl_gt, H_matrix,
                 save_path, idx, print_type, dev, scale_factor,
                 H_predict, is_uni_modal):
   
    batch_size = imgs_search.shape[0]
    line_width = imgs_search.shape[3] // 128
    imgs_search_drawed =  draw_lines(imgs_search, pts_predited,"green",line_width=line_width).to(dev)
    imgs_search_drawed =  draw_lines(imgs_search_drawed, pts_gt,"red",line_width=line_width).to(dev)
    
    imgs_template_expand = torch.ones_like(imgs_search).to(dev)
    
    for b in range(imgs_template_expand.shape[0]):
        h,w = (search_img_size[0]-template_img_size[0])//2, (search_img_size[1]-template_img_size[1])//2
        sar_h,sar_w = template_img_size[0],template_img_size[1]
        imgs_template_expand[b,0,h:h+sar_h,w:w+sar_w] = imgs_template[b,0,:,:]

    gt_mask  =  get_gt_mask(template_img_size=template_img_size,
                            search_img_size = search_img_size,
                            gt_tl= tl_gt,
                            H_matrix = H_matrix, 
                            scale_factor_4_sub_img = 1,
                            stride = 1,
                            device=dev)
    
    gt_mask  =  draw_lines(gt_mask, pts_gt ,"green",line_width=line_width).to(dev)
    gt_mask  =  draw_lines(gt_mask, most_marginal_4_pts_gt,"purple",line_width=line_width).to(dev)
    
    case_save_path = os.path.join(save_path, print_type)
    combine = torch.cat([imgs_search_drawed, gt_mask.expand(-1, 3, -1, -1),imgs_template_expand.expand(-1, 3, -1, -1)],dim=0)
    combined_save_path = os.path.join(case_save_path, f'{idx}.png')
    utils.save_image(combine, combined_save_path, nrow=batch_size, padding=20, pad_value=0.7)
    utils.save_image(imgs_search, os.path.join(case_save_path, f'{idx}_search.png'))
    utils.save_image(imgs_template, os.path.join(case_save_path, f'{idx}_template.png'))
    save_visualize_mix_img(case_save_path, imgs_search, imgs_template, torch.inverse(H_predict), idx, is_uni_modal)
    print(f"++++++++++++++++++++++++++++++++++++++{idx}++++++++++++++++++++++++++++++++++++++++++++++")

def save_visualize_mix_img(save_path, img_search, img_template, H_template_2_search, idx, is_uni_modal):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{idx}_mix.png")
    mix_img = overlay_image_pairs(img_search, img_template,H_template_2_search,is_uni_modal)

    utils.save_image(mix_img, path, nrow=batch_size, padding=20, pad_value=0.7)

def visualization(
    model,
    args,
    dataloader: DataLoader,
    box_parser: get_k_pts_BoxParser,
    size_template,
    stride,
    scale_factor,
    dev: Device,
    save_path,
    config,
    th=5,
):
    model.eval()
    idx = 0

    os.makedirs(save_path, exist_ok=True)

    os.makedirs(os.path.join(save_path, 'normal_case'), exist_ok=True)

    os.makedirs(os.path.join(save_path, 'bad_case'), exist_ok=True)

    with torch.no_grad():
        for imgs_search, imgs_template, tl_gt, H_matrix,imgs_search_rgb, imgs_template_rgb,*_ in tqdm(
            dataloader,
            total=len(dataloader),
            unit="batch",
            desc="[ vis ]",
            colour="blue",
        ):
            idx += tl_gt.shape[0]
            # Transfer data to target device.
            imgs_search = imgs_search.to(dev, non_blocking=True)
            imgs_template = imgs_template.to(dev, non_blocking=True)
            imgs_search_rgb = imgs_search_rgb.to(dev, non_blocking=True)*0.5+0.5
            imgs_template_rgb = imgs_template_rgb.to(dev, non_blocking=True)*0.5+0.5
            
            tl_gt = tl_gt.to(dev, non_blocking=True) #shape[1,2]
            H_matrix = H_matrix.to(dev, non_blocking=True)
            gt_pts,k_pts_in_template = get_4_pts_in_template_and_search_img(tl_gt=tl_gt,
                                                 size_template = size_template,
                                                 H_matrix=H_matrix,
                                                 inner_dis=config['inner_dis'],
                                                 scale_factor=1)
            
            k_pts_in_template = k_pts_in_template.to(dev, non_blocking=True)
            gt_pts = gt_pts.to(dev, non_blocking=True)
            
            template_img_size = (imgs_template.shape[2],imgs_template.shape[3])
            search_img_size = (imgs_search.shape[2],imgs_search.shape[3])

            pred_sim_matrix,pred_score_map, pred_offset_map,x_search,x_template= model(imgs_search, imgs_template)

            pts_pred = box_parser(pred_score_map, pred_offset_map)

            H_predict =  kornia.geometry.transform.get_perspective_transform(pts_pred[:,:,[1,0]], k_pts_in_template[:,:,[1,0]]).to(dev, non_blocking=True)
            
            l2_error, _, m_ = get_l2_error_in_and_out(pts_gt=gt_pts,pts_pred=pts_pred,search_size=imgs_search.shape[-2:])
            most_marginal_4_pts_gt = get_most_marginal_conner_pts(H_matrix,template_img_size)

            
            if l2_error.mean().item() > th :
                save_img_func(imgs_search = imgs_search_rgb, 
                              imgs_template = imgs_template_rgb, 
                              pts_predited=pts_pred, 
                              pts_gt=gt_pts, 
                              most_marginal_4_pts_gt=most_marginal_4_pts_gt, 
                              search_img_size=search_img_size, 
                              template_img_size=template_img_size, 
                              tl_gt=tl_gt, 
                              H_matrix=H_matrix,
                              save_path=save_path, 
                              idx=idx, 
                              print_type='bad_case', 
                              dev=dev, 
                              scale_factor=scale_factor,
                              H_predict=H_predict,
                              is_uni_modal=args.uni_modality)

            else:
                save_img_func(imgs_search = imgs_search_rgb, 
                              imgs_template = imgs_template_rgb, 
                              pts_predited=pts_pred, 
                              pts_gt=gt_pts, 
                              most_marginal_4_pts_gt =most_marginal_4_pts_gt, 
                              search_img_size=search_img_size, 
                              template_img_size=template_img_size, 
                              tl_gt=tl_gt, 
                              H_matrix=H_matrix,
                              save_path=save_path, 
                              idx=idx, 
                              print_type='normal_case', 
                              dev=dev, 
                              scale_factor=scale_factor,
                              H_predict=H_predict,
                              is_uni_modal=args.uni_modality)

def cal_time(
    model,
    dataloader: DataLoader,
    dev: Device,
    args,
):
    model.eval()
    idx = 0
    inference_time = 0.0
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            unit="batch",
            desc="[ val ]",
            colour="blue",
        ):
           
            
            # Transfer data to target device.
            imgs_search = batch[0].to(dev, non_blocking=True)
            imgs_template = batch[1].to(dev, non_blocking=True)
            tl_gt = batch[2].to(dev, non_blocking=True) #shape[1,2]
            H_matrix = batch[3].to(dev, non_blocking=True)
            start_time = time.time()
            idx += tl_gt.shape[0]
            with torch.no_grad():
                pred_sim_matrix,pred_score_map, pred_offset_map,x_search,x_template= model(imgs_search, imgs_template)

            end_time = time.time()

            inference_time += (end_time - start_time)

        
        inference_time /= idx
        inference_time /= args.batch_size
        print(inference_time)
        if torch.cuda.is_available():
            max_memory_allocated = torch.cuda.max_memory_allocated(dev) / 1024**2  # 转换为MB
            print(f"Max memory allocated during inference: {max_memory_allocated:.2f} MB")

def load_dataset(args, config):
    """加载验证数据集"""
    args.batch_size = 1
    validation_loader, *_ = get_val_dataloder(config, args, split=args.split)
    return validation_loader

def print_config(config, indent=0):
    """
    递归打印配置字典，支持嵌套字典
    每个顶级键占一行，子字典的键值对缩进显示
    """
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(' ' * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            logger.info(' ' * indent + f"{key}: {value}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("P2WNET", args.save_dir, 0)

    # Load model config and hyperparameters config.
    with open(args.config, "r") as f:
        config = json.load(f)

    merge_args_into_config(args, config)

    logger.info('=================================================================================')
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info('=================================================================================')
    print_config(config)
    logger.info('=================================================================================')

    # Try to use GPU.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare stateful traning modules.
    epoch: int = 0
    batch_size: int = args.batch_size
    num_epochs: int = config["epochs"]

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

    # 加载checkpoint
    path_checkpoint = args.resume
    if path_checkpoint and path_checkpoint.exists():
        checkpoint = torch.load(path_checkpoint, map_location=dev)
        model_state = checkpoint["model_state"]
        
        # 获取当前模型的state_dict
        current_state = model.state_dict()
        
        # 创建新的state_dict，只包含匹配的层
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

    if args.exe_visualization:
        validation_loader = load_dataset(args, config)
        visualization(
                    model=model,
                    args=args,
                    dataloader=validation_loader,
                    stride=config["downsample_factor"],
                    box_parser=box_parser,
                    size_template=config['training_template_img_size'],
                    dev=dev,
                    scale_factor = config['scale_factor'],
                    save_path = args.save_dir,
                    th=args.threshold,
                    config=config,
                )
        
    if args.exe_statistic:
        validation_loader = load_dataset(args, config)
    
        cal_time(
                model=model,
                dataloader=validation_loader,
                dev=dev,
                args = args,
                )
        
   
