from argparse import ArgumentParser
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
    "--config", "-c",
    type=Path,
    default="configs/general.json"
    )
    parser.add_argument(
        "--batch_size", "-bs",
        type=int,
        default=8
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )
    parser.add_argument(
        "--resume", "-r",
        type=Path,
        nargs='?'
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default="checkpoints"
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=100
    )
    parser.add_argument(
        "--num_of_task_token",
        type=int,
        default=4
    )
    parser.add_argument(
        "--vis_period",
        type=int,
        default=1
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        '--att_scales', 
        type=int, 
        nargs='+', 
        default=[3,5]
    )
    parser.add_argument(
        "--num_features", 
        type=int,
        default=256
    )
    parser.add_argument(
        "--use_share_encoder", 
        action="store_true"
    )
    parser.add_argument(
        "--is_retransformation", 
        action="store_true",
    )
    parser.add_argument(
        '--retransformation_range', 
        type=int, 
        nargs='+', 
        default=[-16,16]
    )
  
    parser.add_argument(
        "--not_load_epoch", 
        action="store_true",
    )
    parser.add_argument(
        "--uni_modality", 
        action="store_true",
    )
    parser.add_argument(
        "--is_recomposition", 
        action="store_true",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=256
    )
    parser.add_argument(
        "--inner_dis",
        type=int,
        default=0
    )
    parser.add_argument(
        "--num_of_predited_pts",
        type=int,
        default=4
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        default="v0"
    )
    parser.add_argument(
        "--box_parser_name",
        default="models.box_parser.get_k_pts_BoxParser",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        default="models.network.MatchModelv15_Multiscale_linear_attn",
        type=str,
    )
    parser.add_argument(
        "--train_fn_name",
        default="train.train_fn.train_one_epoch",
        type=str,
    )
    parser.add_argument(
        "--val_fn_name",
        default="train.train_fn.vaildation",
        type=str,
    )
    parser.add_argument(
        "--loss_fn_name",
        default="train.train_fn.vaildation",
        type=str,
    )
    parser.add_argument(
        "--train_k_epoches_fn_name",
        default="train.train_fn.train_n_epoches_clsreg_version",
        type=str,
    )
    parser.add_argument(
        "--is_val",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--val_period",
        type=int,
        default=1
    )
    parser.add_argument(
        "--x_flip", 
        type=float,
        default=0
    )
    parser.add_argument(
        "--y_flip", 
        type=float,
        default=0
    )
        
    parser.add_argument(
        "--imgs_color",
        type=str,
        default="rgb"
    )
    parser.add_argument(
        "--training_search_img_size",
        type=int,
        nargs=2,
        default=[192, 240]
    )
    parser.add_argument(
        "--training_template_img_size",
        type=int,
        nargs=2,
        default=[64, 80]
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=4
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--lr_decrease_period",
        type=int,
        default=9999
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1
    )
    parser.add_argument(
        "--init_reg_factor",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--reg_factor_increment",
        type=float,
        default=0.4
    )
    parser.add_argument(
        "--offset_learning_ascending_interval",
        type=int,
        default=100
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4
    )
    parser.add_argument(
        "--attention_layers_num",
        type=int,
        default=4
    )
    parser.add_argument(
        "--max_shape",
        type=int,
        nargs=2,
        default=[256, 256]
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=4
    )
    parser.add_argument(
        "--head_kernel_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_backbone_layes",
        type=int,
        default=0
    )
    parser.add_argument(
        "--num_head_layers",
        type=int,
        default=0
    )
    parser.add_argument(
        "--use_self_attention_before_predition",
        action="store_true"
    )
    parser.add_argument(
        "--cnn_norm",
        type=str,
        default="instancenorm"
    )
    parser.add_argument(
        "--cnn_conv2d_has_bias",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--cnn_bn_has_bias",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--cal_gradient",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--attention_drop_out_rate",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--att_activation",
        type=str,
        default="relu"
    )
    parser.add_argument(
        "--cnn_activation",
        type=str,
        default="relu"
    )
    parser.add_argument(
        "--att_norm",
        type=str,
        default="instancenorm"
    )
    parser.add_argument(
        "--att_conv2d_has_bias",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--att_bn_has_bias",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--alpha_focal",
        type=float,
        default=0.25
    )
    parser.add_argument(
        "--gamma_focal",
        type=float,
        default=2
    )
    parser.add_argument(
        "--positive_sample_scale_factor",
        type=float,
        default=0.05
    )
    
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean"
    )
    parser.add_argument(
        "--alpha_giou",
        type=float,
        default=1
    )
    parser.add_argument(
        "--kernel_fn", 
        type=str,
        default='elu',
    )
    parser.add_argument(
        "--is_rescale_template_img", 
        action="store_true",
    )
    parser.add_argument(
        "--lambda_loss_offset", 
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--num_features_predition_head",
        type=int,
        default=384
    )
    parser.add_argument(
        "--min_overlap_ratio",
        type=float,
        default=0.4
    )

    parser.add_argument(
        "--dataset_list",
        type=str,
        nargs='+'
    )

    parser.add_argument(
        "--min_scale_diff",
        type=float,
        default=1
    )
    
    parser.add_argument(
        "--max_scale_diff",
        type=float,
        default=4
    )
    
    parser.add_argument(
        "--is_flip_independent",
        action="store_true"
    )
    parser.add_argument(
        "--sh_path",
        type=str,
        default=""
    )

    parser.add_argument(
        "--sh_file_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--margin",
        type=int,
        nargs=2,
        default=[0,0]
    )

    parser.add_argument(
        "--use_fp16", 
        action="store_true"
    )

    parser.add_argument(
        "--dual_softmax_method",  
        type=str,
        default=None,
        choices=[None, 'separate', 'sequential', 'sinkhorn']
    )

    

    
    #-----------------------------------------------常用训练参数-----------------------------------------------------
    parser.add_argument('--seed', type=int, default=42)
   
    #--------------------------------------------------------------------------------------------------------

    parser.add_argument(
        "--exe_visualization",
        action="store_true",
    )

    parser.add_argument(
        "--exe_statistic",
        action="store_true",
    )

    parser.add_argument(
        "--exe_time",
        action="store_true",
    )

    parser.add_argument(
        "--cal_time",
        action="store_true",
        help="是否执行推理时间测试"
    )


    parser.add_argument(
        "--col_num", "-col_num",
        type=int,
        default=4
    )

    parser.add_argument(
        "--row_num", "-row_num",
        type=int,
        default=4
    )

    parser.add_argument(
        "--display_threshold", 
        type=int,
        default=15
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=10
    )
    parser.add_argument(
        "--inter_domin_classfied_dataset_folder",
        type=str,
        default=""
    )

    parser.add_argument(
        "--exe_inter_domin_classfied",
        action="store_true"
    )

    parser.add_argument(
        "--outer_domin_classfied_dataset_folder",
        type=str,
        default=""
    )
    parser.add_argument(
        "--exe_outer_domin_classfied",
        action="store_true"
    )

    parser.add_argument(
        "--eval_loop_times",
        type=int,
        default=1
    )

    parser.add_argument(
        "--search_image_path",
        type=str,
        default=""
    )
    
    parser.add_argument(
        "--template_image_path",
        type=str,
        default=""
    )
    
    parser.add_argument(
        "--anno_file_path",
        type=str,
        default=""
    )

    parser.add_argument(
        "--img_search_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--img_template_path",  
        type=str,
        default=""
    )

    parser.add_argument('--lr_schedule_type', type=str, default='step',
                        choices=['warmup_cosine', 'step', 'none'],
                        help='学习率调度器类型: warmup_cosine(预热+余弦), step(阶梯衰减), none(恒定)')

    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='预热轮数（仅warmup_cosine类型需要）')

    parser.add_argument('--warmup_start_factor', type=float, default=0.01,
                        help='预热起始学习率因子（仅warmup_cosine类型需要）')

    parser.add_argument('--eta_min', type=float, default=1e-7,
                        help='最小学习率（仅warmup_cosine类型需要）')

    parser.add_argument('--lr_gamma', type=float, default=0.7,
                        help='学习率衰减系数（仅step类型需要）')

    parser.add_argument(
        "--project_name",
        type=str,
        default="SA-HOMO"
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default="none"
    )

    parser.add_argument("--local-rank", type=int, default=0)

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment"
    )

    args = parser.parse_args()
    return args
