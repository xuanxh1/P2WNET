export CUDA_VISIBLE_DEVICES="7"

config_path=configs/general.json
save_path=[your save path]
model=[your ckpt path]


python test_folder/icme_statistic_visualization.py \
  --save_dir=$save_path \
  --config=$config_path \
  --num_workers=8 \
  --num_features=256 \
  --d_model=256 \
  --num_features_predition_head 256 \
  --downsample_factor=32 \
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
  --dataset_list [your dataset path , which contain train and val subfolders] \
  --attn_type "MLA" \
  --scale_factor 1 \
  --inner_dis 0 \
  --seed 42 \
  --uni_modality \
  --model_name "models.network.P2WNET" \
  --resume $model \
  --exe_visualization \
  