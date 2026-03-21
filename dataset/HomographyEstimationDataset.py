import sys
import os
import cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms 
import torch
import numpy as np
from dataset.dataset_utils import get_4_pts, crop_valid_region
import kornia

def random_flip_and_adjust_H_matrix(img1, img2, H_matrix, p_horizontal=0.5, p_vertical=0.5):
    """
    对img1和img2进行同步的水平翻转和垂直翻转，并更新对应的单应性矩阵H
    
    参数:
    img1: tensor, 形状为 [C, H1, W1]
    img2: tensor, 形状为 [C, H2, W2]
    H_matrix: tensor, 形状为 [3, 3], 从img1到img2的映射
    p_horizontal: float, 水平翻转的概率，默认为0.5
    p_vertical: float, 垂直翻转的概率，默认为0.5
    
    返回:
    augmented_img1: tensor, 增强后的img1
    augmented_img2: tensor, 增强后的img2
    augmented_H: tensor, 更新后的单应性矩阵
    """
    
    # 获取图像尺寸
    _, H1, W1 = img1.shape
    _, H2, W2 = img2.shape
    
    # 根据指定概率决定是否进行水平翻转和垂直翻转
    flip_horizontal = torch.rand(1).item() < p_horizontal
    flip_vertical = torch.rand(1).item() < p_vertical
    
    # 定义img1的四个角点
    corners1 = torch.tensor([[0, 0], [W1-1, 0], [0, H1-1], [W1-1, H1-1]], 
                            dtype=H_matrix.dtype, device=H_matrix.device)
    
    # 使用H_matrix计算img2中对应的点
    corners2 = kornia.geometry.transform_points(H_matrix.unsqueeze(0), corners1.unsqueeze(0)).squeeze(0)
    
    # 对图像进行翻转
    if flip_horizontal:
        img1 = torch.flip(img1, [2])  # 水平翻转
        img2 = torch.flip(img2, [2])  # 水平翻转
        corners1[:, 0] = W1 - 1 - corners1[:, 0]
        corners2[:, 0] = W2 - 1 - corners2[:, 0]
    
    if flip_vertical:
        img1 = torch.flip(img1, [1])  # 垂直翻转
        img2 = torch.flip(img2, [1])  # 垂直翻转
        corners1[:, 1] = H1 - 1 - corners1[:, 1]
        corners2[:, 1] = H2 - 1 - corners2[:, 1]
    
    # 使用Kornia的get_perspective_transform计算新的H矩阵
    augmented_H = kornia.geometry.transform.get_perspective_transform(corners1.unsqueeze(0), corners2.unsqueeze(0)).squeeze(0)
    
    return img1, img2, augmented_H

def generate_homo(img1, img2, homo_parameter):
    # 处理marginal参数，支持单一值或(x,y)分离值
    marginal = homo_parameter["marginal"]

    marginal_x, marginal_y = marginal
    perturb, patch_size = homo_parameter["perturb"], homo_parameter["patch_size"]
    height, width = homo_parameter["height"], homo_parameter["width"]
    
    # 使用分离的marginal值来随机选择patch位置
    x = random.randint(marginal_x, width - marginal_x - patch_size[1])
    y = random.randint(marginal_y, height - marginal_y - patch_size[0])

    top_left = (x, y)
    bottom_left = (x, patch_size[0] + y - 1)
    bottom_right = (patch_size[1] + x - 1, patch_size[0] + y - 1)
    top_right = (patch_size[1] + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    
    # 使用分离的marginal值来裁剪图像
    img1 = img1[top_left[1]-marginal_y:bottom_right[1]+marginal_y+1, 
                top_left[0]-marginal_x:bottom_right[0]+marginal_x+1, :] 
    img2 = img2[top_left[1]-marginal_y:bottom_right[1]+marginal_y+1, 
                top_left[0]-marginal_x:bottom_right[0]+marginal_x+1, :] 

    # 调整坐标系，将top_left设置为(marginal_x, marginal_y)
    four_pts = four_pts - four_pts[np.newaxis, 0] + np.array([marginal_x, marginal_y])
    (top_left, top_right, bottom_left, bottom_right) = four_pts
    
    perturb_x, perturb_y = perturb
    try:
        four_pts_perturb = []
        for i in range(4):
            t1 = random.randint(-perturb_x, perturb_x)
            t2 = random.randint(-perturb_y, perturb_y)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    except:
        four_pts_perturb = []
        for i in range(4):
            t1 =   perturb_x // (i + 1)
            t2 = - perturb_y // (i + 1)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    
    warped_img1 = cv2.warpPerspective(img1, H_inverse, (img1.shape[1], img1.shape[0]))
    if(warped_img1.ndim==2):
        warped_img1 = warped_img1[..., np.newaxis]
    
    patch_img1 = warped_img1[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :] 
    patch_img2 = img2[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :] 

    new_dst_pts = np.array(dst_pts - np.array([marginal_x, marginal_y]), dtype=np.float32)
    H_patch_img2_to_warped_img1 = cv2.getPerspectiveTransform(new_dst_pts, org_pts)

    new_org_pts = np.array(org_pts - np.array([marginal_x, marginal_y]), dtype=np.float32)
    H_patch_img2_to_patch_img1 = cv2.getPerspectiveTransform(new_dst_pts, new_org_pts)

    # 优化：一次性转换所有numpy数组为tensor，减少重复转换开销
    ground_truth = torch.from_numpy(ground_truth).to(torch.float32)
    org_pts = torch.from_numpy(org_pts).to(torch.float32)
    dst_pts = torch.from_numpy(dst_pts).to(torch.float32)

    """
    img1中dst_pts在warped_img1中是org_pts

    new_H_inverse代表的是从patch_img2到warped_img1的映射
    """
    return patch_img1, patch_img2, ground_truth, org_pts, dst_pts, warped_img1, img2 , H_patch_img2_to_warped_img1, H_patch_img2_to_patch_img1

class Homography_Dataset_runtime(Dataset):
    def __init__(self, 
                 root_list,  # Modified to accept a list of dataset paths
                 split='train',  # Added split parameter
                 search_size=(768,960), 
                 template_patch_size=(256,320),
                 transform=transforms.ToTensor(), 
                 x_flip=0, 
                 y_flip=0,
                 color='gray',
                 uni_modality=False,
                 min_overlap_ratio=1,
                 ) -> None:
        super().__init__()
        """
        input_template_size (h,w)
        """
        self.search_size = search_size
        self.template_patch_size = template_patch_size
        self.transform = transform
        self.x_flip = x_flip
        self.min_overlap_ratio = min_overlap_ratio
        self.y_flip = y_flip
        self.uni_modality = uni_modality
        self.homo_parameter = {"marginal":(0,0), "perturb":(search_size[0] // 4, search_size[1] // 4), "patch_size":search_size}
        self.color = color

        # Initialize lists to store image paths from all datasets
        self.template_imgs_paths = []
        self.search_imgs_paths = []
        self.dataset_indices = []
        
        # Iterate through all dataset paths
        for dataset_idx, root in enumerate(root_list):
            template_folder = os.path.join(root, split, 'template')
            search_folder = os.path.join(root, split, 'search')
            
            if not os.path.exists(template_folder) or not os.path.exists(search_folder):
                print(f"Warning: {template_folder} or {search_folder} does not exist, skipping...")
                continue
                
            template_imgs_names = sorted(os.listdir(template_folder))
            search_imgs_names = sorted(os.listdir(search_folder))
            
            # Ensure template and search image counts match
            if len(template_imgs_names) != len(search_imgs_names):
                print(f"Warning: Number of template and search images do not match in {root}, skipping...")
                continue
                
            # Add full paths to lists
            for t_name, s_name in zip(template_imgs_names, search_imgs_names):
                self.template_imgs_paths.append(os.path.join(template_folder, t_name))
                self.search_imgs_paths.append(os.path.join(search_folder, s_name))
                self.dataset_indices.append(dataset_idx)
                
        print(f"Loaded {len(self.template_imgs_paths)} samples from {len(root_list)} datasets")

    def __len__(self):
        return len(self.search_imgs_paths)

    def crop_patch_from_trans_template(self, transformed_image, H, search_size,crop_size):
         #this case do not need crop
        if crop_size[0]==self.search_size[1] and crop_size[1]==self.search_size[0]:
            gt_tl = torch.tensor([0, 0],dtype=torch.float32)
            return transformed_image , gt_tl

        template_croped,tl = crop_valid_region(transformed_image, H,search_size, crop_size,min_overlap_ratio=self.min_overlap_ratio)

        if  template_croped is not None and tl is not None:
            gt_tl = torch.tensor([tl[1],tl[0]],dtype = torch.float)
            return template_croped , gt_tl
        else:
            w, h, _ = transformed_image.shape
            center_x, center_y = (w-crop_size[0])//2 , (h - crop_size[1])//2
            template_croped = transformed_image[center_y:center_y + crop_size[1], center_x : center_x + crop_size[0]]
            gt_tl = torch.tensor([center_y,center_x],dtype = torch.float)
            return template_croped , gt_tl

    def synchronized_resize_cv2(self, template_img, search_img, target_size):
        """
        Synchronously crop and resize two images using OpenCV
        """
        # Get original dimensions
        h, w = search_img.shape[:2]  # CV2 image shape returns (height, width)
        target_h, target_w = target_size
        
        # Resize to target size
        template_img = cv2.resize(template_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        search_img = cv2.resize(search_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        return template_img, search_img

    def __getitem__(self, i):
        template_img_path = self.template_imgs_paths[i]
        search_img_path = self.search_imgs_paths[i]

        if self.uni_modality:
            template_img_path = search_img_path
        
        template_img = cv2.imread(template_img_path, cv2.IMREAD_COLOR)
        search_img = cv2.imread(search_img_path, cv2.IMREAD_COLOR)
        # Convert from BGR to RGB
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
   
        h, w = search_img.shape[:2]
        

        h_s,w_s = self.search_size
        h_c,w_c = self.template_patch_size
        if self.search_size !=(h,w):
            # Use OpenCV-based synchronized resize
            template_img, search_img = self.synchronized_resize_cv2(template_img, search_img, (h_s, w_s))
        
        self.homo_parameter["height"], self.homo_parameter["width"] = self.search_size
        patch_img1, _,_, _,_, _,_,H_matrix,_ = generate_homo(template_img, search_img, homo_parameter=self.homo_parameter)
        template_croped , gt_tl = self.crop_patch_from_trans_template(patch_img1, H_matrix, (w_s,h_s), (w_c,h_c))

        H_matrix = torch.from_numpy(H_matrix).to(torch.float)

        dst_pts, org_pts = get_4_pts(gt_tl.unsqueeze(dim=0), self.template_patch_size, H_matrix.unsqueeze(dim=0), inner_dis=0, scale_factor=1)

        org_pts = torch.tensor([[0, 0],
                                [0, w_c-1],
                                [h_c-1, 0],
                                [h_c-1, w_c-1]], dtype=torch.float32).unsqueeze(dim=0)
        
        H_matrix_new = kornia.geometry.transform.get_perspective_transform(dst_pts[:,:,[1,0]],org_pts[:,:,[1,0]]).squeeze(dim=0)
        gt_tl = torch.tensor([0, 0],dtype=torch.float32)

        if self.transform:
            template_croped = Image.fromarray(template_croped)
            search_img = Image.fromarray(search_img)
            template_croped_rgb = template_croped.convert('RGB')
            search_img_rgb = search_img.convert('RGB')
            if self.color == 'gray':
                template_croped = template_croped.convert('L')
                search_img = search_img.convert('L')
            
            template_croped = self.transform(template_croped)
            search_img = self.transform(search_img)
            template_croped_rgb = self.transform(template_croped_rgb)
            search_img_rgb = self.transform(search_img_rgb)
        
        if self.x_flip != 0 or self.y_flip != 0:
            search_img, template_croped, H_matrix_new = random_flip_and_adjust_H_matrix(search_img, template_croped, H_matrix_new, self.x_flip, self.y_flip)
        
        return search_img, template_croped, gt_tl, H_matrix_new,search_img_rgb,template_croped_rgb,search_img_path,template_img_path

