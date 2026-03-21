import torch
import torch.nn as nn


class get_k_pts_BoxParser(nn.Module):
    def __init__(self, stride: int) -> None:
        super(get_k_pts_BoxParser, self).__init__()
        self.stride = stride

    @torch.no_grad()
    def forward(self, score_map: torch.Tensor, offset_map: torch.Tensor) -> torch.Tensor:
        bs, num_pts, ho, wo = score_map.shape
       
        # Flatten score_map to [bs, num_pts, flat_size]
        score_flat = score_map.flatten(start_dim=2)

        # Get indices of max scores [bs, num_pts]
        indices = score_flat.argmax(dim=2)

        # Reshape offset_map to [bs, num_pts, 2, ho, wo] then flatten to [bs, num_pts, 2, flat_size]
        offset_flat = offset_map.view(bs, num_pts, 2, ho, wo).flatten(start_dim=3)

        # Prepare indices for gathering: [bs, num_pts, 2, 1]
        indices_exp = indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 2, 1)

        # Gather offsets [bs, num_pts, 2, 1]
        offset_gathered = offset_flat.gather(dim=3, index=indices_exp)

        # Squeeze to [bs, num_pts, 2]
        offset = offset_gathered.squeeze(3)

        # Compute coarse coordinates
        y = torch.div(indices, wo, rounding_mode="floor")
        x = indices - y * wo
        tl_downscaled_coarse = torch.stack((y, x), dim=2).float()

        # Refined downscaled
        tl_downscaled = tl_downscaled_coarse + offset

        # Upscale
        tl = tl_downscaled * self.stride

        return tl

    