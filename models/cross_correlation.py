import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class BatchDepthwiseCrossCorrelation(nn.Module):
    def __init__(self):
        """批量深度可分离互相关层"""
        super(BatchDepthwiseCrossCorrelation, self).__init__()
    
    def forward(self, x: Tensor, templates: Tensor) -> Tensor:
        """
        
        Args:
            x:  [bs, nc, hi, wi]
            templates:  [num_templates, bs, nc, ht, wt]
        
        Returns:
            xcorr:  [num_templates, bs, nc, hi, wi]
        """
        num_templates, bs, nc, ht, wt = templates.shape
        _, _, hi, wi = x.shape
        
        # [bs, nc, hi, wi] -> [bs*num_templates, nc, hi, wi]
        x_repeated = x.unsqueeze(1).repeat(1, num_templates, 1, 1, 1)  # [bs, num_templates, nc, hi, wi]
        x_repeated = x_repeated.reshape(bs * num_templates, nc, hi, wi)
        
        # [num_templates, bs, nc, ht, wt] -> [bs*num_templates*nc, 1, ht, wt]
        templates_permuted = templates.permute(1, 0, 2, 3, 4)  # [bs, num_templates, nc, ht, wt]
        kernels = templates_permuted.reshape(bs * num_templates * nc, 1, ht, wt)
        
        # [bs*num_templates, nc, hi, wi] -> [1, bs*num_templates*nc, hi, wi]
        x_input = x_repeated.reshape(1, bs * num_templates * nc, hi, wi)
        
        pad_h = int(ht) // 2
        pad_w = int(wt) // 2
        xcorr = F.conv2d(x_input, kernels, groups=bs * num_templates * nc, padding=(pad_h, pad_w))
        
        # [1, bs*num_templates*nc, hi, wi] -> [bs, num_templates, nc, hi, wi]
        xcorr = xcorr.reshape(bs, num_templates, nc, hi, wi)
        
        # [bs, num_templates, nc, hi, wi] -> [num_templates, bs, nc, hi, wi]
        xcorr = xcorr.permute(1, 0, 2, 3, 4)
        
        return xcorr
    

