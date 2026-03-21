export CUDA_VISIBLE_DEVICES="4,5"


save_path="test_out/p2wnet/train"
config_path=configs/general.json

torchrun --nproc_per_node 2 --master_port=29506 train/MLA_ddp.py \
  --batch_size=4 \
  --lr 1e-4 \
  --save_dir=$save_path \
  --config=$config_path \
  --num_workers=8 \
  --epochs 1000 \
  --save_period=40 \
  --num_features=256 \
  --d_model=256 \
  --num_features_predition_head 256 \
  --downsample_factor=32 \
  --weight_decay=1e-6 \
  --num_of_predited_pts=4 \
  --num_of_task_token=4 \
  --training_search_img_size 1536 1920 \
  --training_template_img_size 512 640 \
  --cnn_activation "gelu" \
  --att_activation "gelu" \
  --kernel_fn "elu" \
  --cnn_norm "instancenorm" \
  --att_norm "instancenorm" \
  --imgs_color "gray" \
  --att_scales 3 5 \
  --attention_layers_num 2 \
  --dataset_list "[your dataset path , which contain train and val subfolders]" \
  --attn_type "MLA" \
  --inner_dis 0 \
  --lambda_loss_offset 1e-3 \
  --seed 42 \
  --model_name "models.network.P2WNET" \
  --val_fn_name "train.train_fn_ddp.vaildation_one_epoch" \
  --train_fn_name "train.train_fn_ddp.train_one_epoch" \
  --train_k_epoches_fn_name "train.train_fn_ddp.train_n_epoches" \
  --loss_fn_name "models.loss_k_pts.Loss_Fn" \
  --lr_schedule_type "step" \




