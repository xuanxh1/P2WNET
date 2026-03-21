from typing import Tuple, Dict
from torch import Tensor
import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.95,is_sofmax=False):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.is_softmax = is_sofmax
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        loss: Tensor = self.bce_criterion(pred, gt)
        if self.is_softmax:
            pred_prob = torch.softmax(pred,dim = 2)
        else:
            pred_prob = torch.sigmoid(pred)  # prob from logits  

        p_t = gt * pred_prob + (1.0 - gt) * (1.0 - pred_prob)
        alpha_factor = gt * self.alpha + (1.0 - gt) * (1.0 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        return loss.mean()

 
class GIoULoss(nn.Module):
    def __init__(self, size_box: Tuple[int, int], stride: int):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        h1, w1 = size_box
        h1_s, w1_s = h1 / stride, w1 / stride

        # Get downscaled box size.
        self.register_buffer(
            "hw_box",
            torch.tensor((h1_s, w1_s), dtype=torch.float32).view(
                (1, 2, 1, 1))  # -> [1, 2, 1, 1]
        )
        self.area_box = h1_s * w1_s

    def forward(self, tl_pred: Tensor, tl_gt: Tensor) -> Tensor:
        tl_min = torch.minimum(tl_pred, tl_gt)  # -> [bs, 2, ho, wo]
        tl_max = torch.maximum(tl_pred, tl_gt)  # -> [bs, 2, ho, wo]

        # Compute intersection area.
        area_inter = (
            self.hw_box - (tl_max - tl_min)
        ).clamp_min(0).prod(dim=1, keepdim=True)  # -> [bs, 1, ho, wo]

        # Compute union area.
        area_union = 2 * self.area_box - \
            area_inter + self.eps  # -> [bs, 1, ho, wo]
        # Compute IoU.
        iou = area_inter / area_union

        # Compute convex bounding box area.
        hw_cbox = tl_max + self.hw_box - tl_min
        area_cbox = hw_cbox.prod(dim=1, keepdim=True) + \
            self.eps  # -> [bs, 1, ho, wo]

        # Compute GIoU loss.
        giou = iou - (area_cbox - area_union) / area_cbox

        loss = 1.0 - giou
    
        return loss        
    
class Loss_Fn(nn.Module):
    def __init__(
        self,
        size_optical: Tuple[int, int], 
        size_sar: Tuple[int, int],
        stride: int,
        config_loss: Dict[str, float],
        
    ) -> None:
        super().__init__()
        self.criterion_score_map = BinaryFocalLoss(
            gamma=config_loss["gamma_focal_loss"],
            alpha=config_loss["alpha_focal_loss"],
            is_sofmax=config_loss["is_sofmax"]
        )
        # self.criterion_score_map = ModifiedFocalLoss()
        self.criterion_offset_map = GIoULoss(size_sar, stride)
        
        self.stride = stride
        # Genrate yx grid on the output scale.
        h0, w0 = size_optical
        h1, w1 = size_sar
        self.sar_h = h1
        self.sar_w = w1
        self.opt_h = h0
        self.opt_w = w0
        ho, wo = h0  // stride , w0 // stride 
        y = torch.arange(ho, dtype=torch.float32)  # -> [ho]
        x = torch.arange(wo, dtype=torch.float32)  # -> [wo]
        grid_y, grid_x = torch.meshgrid(y, x)  # -> [ho, wo], [ho, wo]
        self.register_buffer(
            "grid_yx",
            # -> [1, 2, ho, wo]
            torch.stack((grid_y, grid_x), dim=0).unsqueeze(dim=0)
        )

        self.radius_pos = \
            0.5 * config_loss["positive_sample_scale_factor"] \
                * max(*size_sar) / stride

        self.radius_pos = max(0.5,self.radius_pos)
        self.lambda_loss_offset = config_loss["lambda_loss_offset"]
    
    def get_k_pts_mask_pos(self, gt_pts):
        """
        gt_pts: Tensor of shape [bs, k, 2]
        
        Returns:
        mask_pos: Tensor of shape [bs, k, ho, wo]
        """
        bs, k, _ = gt_pts.shape
        valid_points = ((gt_pts[..., 0] >= 0) & (gt_pts[..., 0] < self.opt_h) & 
                    (gt_pts[..., 1] >= 0) & (gt_pts[..., 1] < self.opt_w))  # [bs, k]
        
        valid_mask = valid_points.unsqueeze(-1).unsqueeze(-1).float()  # [bs, k, 1, 1]

        tl_gt_downscaled = gt_pts.view((bs, k, 2, 1, 1)) / self.stride
        #使得若tl_gt不在合理的范围内， 则将其修改到合理的范围内中的最近点

        grid_offset = self.grid_yx.unsqueeze(1) - (tl_gt_downscaled.floor() + 0.5)

        grid_dist, _ = grid_offset.abs().max(dim=2,keepdim=False)  # -> [bs, k, 1, ho, wo]
        
        mask_pos = (grid_dist <= self.radius_pos + 0.001)  # -> [bs, k, 1, ho, wo]

        mask_pos = mask_pos * valid_mask

        return mask_pos

    def get_k_pts_gt_score_map(self, gt_pts):
        """
        gt_pts: Tensor (bs, k, 2)
        
        return 
        
        score_map_gt (bs, k, h, w)
        """
        
        score_map_gt = self.get_k_pts_mask_pos(gt_pts).to(torch.float32)
         
        return score_map_gt

    def get_k_pts_reg_loss(self, offset_map_pred, pts):
        """
        Computes the regression loss for k points in a vectorized manner,
        matching the original loop's summation logic.
        offset_map_pred [bs, 2k, h0, w0]
        pts [bs, k, 2]
        """
        bs, k, _ = pts.shape
        h, w = offset_map_pred.shape[-2:] # Get spatial dimensions h0, w0

        # 1. Prepare predicted points [bs, 2k, h, w] -> [bs*k, 2, h, w]
        pts_pred_downscaled = self.grid_yx.expand(bs, -1, -1, -1).repeat(1, k, 1, 1) + offset_map_pred
        pts_pred_reshaped = pts_pred_downscaled.view(bs * k, 2, h, w)

        # 2. Prepare ground truth points [bs, k, 2] -> [bs*k, 2, 1, 1]
        pts_gt_downscaled = pts / self.stride
        pts_gt_reshaped = pts_gt_downscaled.reshape(bs * k, 2).unsqueeze(-1).unsqueeze(-1)

        # 3. Calculate loss map for all points simultaneously
        # Expecting criterion to handle [N, C, H, W] vs [N, C, 1, 1] -> [N, 1, H, W] or similar
        loss_offset_map_all: Tensor = self.criterion_offset_map(
            pts_pred_reshaped, pts_gt_reshaped
        )  # -> [bs*k, 1, h, w]

        # Reshape loss map back: [bs*k, 1, h, w] -> [bs, k, h, w]
        # Squeeze the channel dim 1 if it exists from the criterion
        loss_offset_map_grouped = loss_offset_map_all.view(bs, k, -1, h, w).squeeze(2) # -> [bs, k, h, w]

        # 4. Get masks for all points [bs, k, h, w]
        mask_pos_all = self.get_k_pts_gt_score_map(pts) # -> [bs, k, h, w]
        # Ensure mask is float for multiplication
        mask_pos_all = mask_pos_all.float()

        # 5. Calculate weighted loss (element-wise)
        weighted_loss = loss_offset_map_grouped * mask_pos_all # [bs, k, h, w]

        # 6. Sum weighted loss and mask FOR EACH k ACROSS ALL OTHER DIMS (bs, h, w)
        # This matches the original loop's .sum() behavior inside the loop
        sum_weighted_loss_per_k = weighted_loss.sum(dim=[0, 2, 3]) # Sum over bs, h, w -> shape [k]
        sum_mask_per_k = mask_pos_all.sum(dim=[0, 2, 3])         # Sum over bs, h, w -> shape [k]

        # 7. Calculate normalized loss per k, avoiding division by zero
        # Create a tensor of zeros with the same shape [k], dtype, and device
        zeros = torch.zeros_like(sum_mask_per_k)
        loss_offset_per_k = torch.where(
            sum_mask_per_k > 1e-8,                     # Condition (use epsilon for float comparison)
            sum_weighted_loss_per_k / (sum_mask_per_k + 1e-8), # Value if True (add epsilon for stability)
            zeros                                      # Value if False
        ) # -> shape [k]

        # 8. Sum the normalized losses across all points k
        reg_loss = loss_offset_per_k.sum() # -> scalar loss

        return reg_loss

    def forward(
        self,
        pred: Tuple[Tensor, Tensor],
        pts: Tensor,
        reg_factor: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        pred = (score_map_pred, offset_map_pred)
        score_map_pred [bs,k,h0,w0]
        offset_map_pred [bs,2k,h0,w0]
        pts [bs,k,2]
        """
        score_map_pred, offset_map_pred = pred
       
        get_k_pts_gt_score_map = self.get_k_pts_gt_score_map(pts)
        
        loss_cls = self.criterion_score_map(score_map_pred, get_k_pts_gt_score_map)
        
        loss_offset = self.get_k_pts_reg_loss(offset_map_pred,pts)
         
        loss_total = loss_cls + self.lambda_loss_offset * loss_offset
       
        return loss_cls, loss_offset, loss_total

class sim_matrix_loss_general(nn.Module):
    def __init__(
        self,
        size_optical,
        size_sar,
        stride,
        config_loss,
        scale_factor=1,
    ) -> None:   
        super().__init__()
        self.h = size_optical[0] * size_optical[1] // (stride ** 2)
        self.w = size_sar[0] * size_sar[1] // (stride ** 2)
        self.sar_h = size_sar[0] // stride
        self.sar_w = size_sar[1] // stride
        self.optical_h = size_optical[0] // stride
        self.optical_w = size_optical[1] // stride
        self.stride = stride

        meshgrid = torch.meshgrid(torch.arange(self.sar_h), torch.arange(self.sar_w), indexing='ij')
        grid_x = meshgrid[0] * stride // scale_factor
        grid_y = meshgrid[1] * stride // scale_factor
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)
        
        self.criterion_score_map = BinaryFocalLoss(
            gamma=config_loss["gamma_focal_loss"],
            alpha=1 - 1 / self.w
        )

    def forward(self, pred, gt, H):
        batch_size = gt.shape[0]
        device = gt.device
        gt_sim_matrix = torch.zeros(batch_size, self.h, self.w, device=device)

        grid_x = self.grid_x.repeat(batch_size, 1, 1)
        grid_y = self.grid_y.repeat(batch_size, 1, 1)
        grid_x = grid_x + gt[:, 0].unsqueeze(1).unsqueeze(2)
        grid_y = grid_y + gt[:, 1].unsqueeze(1).unsqueeze(2)

        H_inv = torch.inverse(H)

        ones = torch.ones_like(grid_x)
        coords = torch.stack([grid_y, grid_x, ones], dim=-1).to(torch.float32)
        coords_transformed = torch.matmul(coords.reshape(batch_size, -1, 3), H_inv.transpose(-1, -2))
        coords_transformed = coords_transformed.reshape(batch_size, self.sar_h, self.sar_w, 3)
        
        coords_transformed[..., :2] /= coords_transformed[..., 2:3]

        indices_h = (coords_transformed[..., 1] // self.stride).long()
        indices_w = (coords_transformed[..., 0] // self.stride).long()
        
        # 计算组件级有效掩码
        valid_mask_flat = (
            (indices_h.reshape(-1) >= 0) &
            (indices_h.reshape(-1) < self.optical_h) &
            (indices_w.reshape(-1) >= 0) &
            (indices_w.reshape(-1) < self.optical_w)
        ).to(device)

        batch_indexes = torch.arange(batch_size).view(-1, 1).expand(-1, self.w).reshape(-1).to(device)[valid_mask_flat]
        h_indexes = ((indices_h.reshape(-1) * self.optical_w + indices_w.reshape(-1)).long().to(device))[valid_mask_flat]
        w_indexes = torch.arange(self.w).repeat(batch_size).to(device)[valid_mask_flat]

        gt_sim_matrix[batch_indexes, h_indexes, w_indexes] = 1
        sim_matrix_loss = self.criterion_score_map(pred, gt_sim_matrix)

        return sim_matrix_loss


    
    