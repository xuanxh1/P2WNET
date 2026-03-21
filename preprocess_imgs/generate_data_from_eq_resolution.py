import sys
sys.path.append('/data/xieshangxuan/master/img_matching/he_2025_11_30') 
import cv2
import numpy as np
import math
import os
import shutil
from models.common import get_gt_sim_matrix_v1
import torch
from torchvision import utils
from models.homography_transformed import get_gt_mask
from PIL import Image
from torchvision import transforms 
from dataset.dataset_for_homo_with_dynamic_homo_trans import generate_homo
from dataset.dataset_utils import get_4_pts, crop_valid_region
import kornia
import torchvision.utils as vutils


def crop_patch_from_trans_template( transformed_image, H, search_size,crop_size):
        #this case do not need crop
    if crop_size[0]==search_size[0] and crop_size[1]==search_size[1]:
        gt_tl = torch.tensor([0, 0],dtype=torch.float32)
        return transformed_image , gt_tl

    template_croped,tl = crop_valid_region(transformed_image, H,search_size, crop_size)

    if  template_croped is not None and tl is not None:
        gt_tl = torch.tensor([tl[1],tl[0]],dtype = torch.float)
        return template_croped , gt_tl
    else:
        w, h = transformed_image.shape
        center_x, center_y = (w-crop_size[0])//2 , (h - crop_size[1])//2
        template_croped = transformed_image[center_y:center_y + crop_size[1], center_x : center_x + crop_size[0]]
        gt_tl = torch.tensor([center_y,center_x],dtype = torch.float)
        return template_croped , gt_tl

def gen_data(img_folder_path, output_folder_path, crop_size, warp_num=1):
    """
    对opt对应的sar图片, 进行单应性变换,并同时返回相似矩阵,与真值mask
    """
    search_img_folder = os.path.join(img_folder_path, 'search')
    template_img_folder = os.path.join(img_folder_path, 'template')
    search_img_names = sorted(os.listdir(search_img_folder))
    template_img_names = sorted(os.listdir(template_img_folder))

    os.makedirs(output_folder_path, exist_ok=True)
    folder_for_search_img = os.path.join(output_folder_path, 'search')
    os.makedirs(folder_for_search_img, exist_ok=True)
    folder_for_template_img = os.path.join(output_folder_path, 'template')
    os.makedirs(folder_for_template_img, exist_ok=True)
    folder_for_img_after_homography_transformed = os.path.join(output_folder_path, 'homography_transformed')
    os.makedirs(folder_for_img_after_homography_transformed, exist_ok=True)
    folder_for_img_after_homography_transformed_croped_original_size = os.path.join(output_folder_path, 'homography_transformed_croped')
    os.makedirs(folder_for_img_after_homography_transformed_croped_original_size, exist_ok=True)

    out_anno_file_path = os.path.join(output_folder_path, 'anno.txt')
    file = open(out_anno_file_path, 'w')
    idx = 0

    for search_name, template_name in zip(search_img_names, template_img_names):
        template_img_path = os.path.join(img_folder_path, 'template', template_name)
        search_img_path = os.path.join(img_folder_path, 'search', search_name)

        # 加载图像并转换为灰度图
        template_img = Image.open(template_img_path).convert('L')
        search_img = Image.open(search_img_path).convert('L')

        # 图像尺寸
        width_search, height_search = search_img.size
        width_crop, height_crop = crop_size

        # 保存灰度化的原始搜索图和模板图
        search_img_save_path = os.path.join(folder_for_search_img, f'{search_name}')
        search_img.save(search_img_save_path)
        template_image_save_path = os.path.join(folder_for_template_img, f'{template_name}')
        template_img.save(template_image_save_path)

        # 转换为NumPy数组并添加通道维度
        template_img = np.array(template_img)[..., np.newaxis]
        search_img = np.array(search_img)[..., np.newaxis]

        # 设置映射参数
        homo_parameter = {"marginal": (0,0), "perturb": (width_search // 4, height_search // 4), "patch_size": (height_search, width_search)}
        homo_parameter["height"], homo_parameter["width"] = height_search, width_search

        for warp_idx in range(warp_num):
            idx += 1
            if idx % 50 == 0:
                print(f'has process {idx} imgs')

            # 生成单应性变换
            template_warpped, patch_img2, four_gt, org_pts, dst_pts, large_img1_warp, large_img2,H_matrix,_= generate_homo(template_img, search_img, homo_parameter=homo_parameter)
            template_croped, gt_tl = crop_patch_from_trans_template(template_warpped, H_matrix, (width_search, height_search), (width_crop, height_crop))

            # 转换为张量
            transform = transforms.ToTensor()
            template_croped = transform(template_croped.squeeze())
            template_warpped = transform(template_warpped.squeeze())

            H_matrix = torch.from_numpy(H_matrix).to(torch.float)
            dst_pts, org_pts = get_4_pts(gt_tl.unsqueeze(dim=0), (height_crop, width_crop), H_matrix.unsqueeze(dim=0), inner_dis=0, scale_factor=1)

            org_pts = torch.tensor([[0, 0],
                                    [0, width_crop-1],
                                    [height_crop-1, 0],
                                    [height_crop-1, width_crop-1]], dtype=torch.float32).unsqueeze(dim=0)
            
            H_matrix_new = kornia.geometry.transform.get_perspective_transform(dst_pts[:, :, [1, 0]], org_pts[:, :, [1, 0]]).squeeze(dim=0)
            gt_tl = torch.tensor([0, 0], dtype=torch.float32)

            transformed_name = template_name.split('.')[0] + f'_warp_idx_{warp_idx+1}.png'
            transformed_image_save_path = os.path.join(folder_for_img_after_homography_transformed, transformed_name)
            vutils.save_image(template_warpped, transformed_image_save_path)

            template_sub_name = template_name.split('.')[0] + f'_warp_idx_{warp_idx+1}.png'
            # 保存 H 矩阵时使用完整精度
            file.write(f'{search_name} {template_sub_name} {0.0} {0.0} {H_matrix_new[0,0]} {H_matrix_new[0,1]} {H_matrix_new[0,2]} {H_matrix_new[1,0]} {H_matrix_new[1,1]} {H_matrix_new[1,2]} {H_matrix_new[2,0]} {H_matrix_new[2,1]} {H_matrix_new[2,2]}\n')
            transformed_croped_original_size_image_save_path = os.path.join(folder_for_img_after_homography_transformed_croped_original_size, f'{template_sub_name}')
            vutils.save_image(template_croped, transformed_croped_original_size_image_save_path)

    file.close()

if __name__ =='__main__':

    warp_num = 1
    img_folder_path1 = '/data/xieshangxuan/master/img_matching/he_2025_11_30/data/training_data/simulation/Landsat8_high_quality_5220/train'
    output_path1  = f'/data/xieshangxuan/master/img_matching/he_2025_11_30/data/training_data/simulation/Landsat8_high_quality_5220_static_{warp_num}_gray_correct_org_pts/train'
    gen_data(img_folder_path1,output_path1,crop_size=(640,512),warp_num=warp_num)


    img_folder_path2 = '/data/xieshangxuan/master/img_matching/he_2025_11_30/data/training_data/simulation/Landsat8_high_quality_5220/val'
    output_path2  = f'/data/xieshangxuan/master/img_matching/he_2025_11_30/data/training_data/simulation/Landsat8_high_quality_5220_static_{warp_num}_gray_correct_org_pts/val'
    gen_data(img_folder_path2,output_path2,crop_size=(640,512),warp_num=warp_num)

