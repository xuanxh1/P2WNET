import copy
import torch.nn as nn
from .multiscale_linear_attention import *

class Multiscale_Linear_attention(nn.Module):
    def __init__(self, 
                 d_model,
                 nhead,
                 layer_names=['self', 'cross'] * 4,
                 attention_drop_out_rate=0.0,
                 scales=(5,),
                 att_act = 'gelu',
                 att_conv2d_has_bias=True,
                 att_bn_has_bias=False,
                 att_norm = 'instancenorm',
                 attn_type='v0',
                 kernel_fn='relu',
                 ):
        super(Multiscale_Linear_attention, self).__init__()

        self.attn_type = attn_type
        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        
        attn_params = {
        'd_model': d_model,
        'nhead': nhead,
        'attention_drop_out_rate': attention_drop_out_rate,
        'act': att_act,
        'scales': scales,
        'conv_has_bias': att_conv2d_has_bias,
        'bn_has_bias': att_bn_has_bias,
        'norm': att_norm,
        'kernel_fn': kernel_fn
        }

        orgv_params = {
            'd_model':d_model, 
            'nhead':nhead, 
            'attention':'linear', 
            'attention_drop_out_rate':0, 
            'act': att_act
        }

        if attn_type=='MLA':
            encoder_layer_class = EncoderLayer_Multiscale_linear
        else:
            raise Exception("attn type incorrect")
        

        if attn_type=='orgv':
            encoder_layer = encoder_layer_class(**orgv_params)
        else:
            encoder_layer = encoder_layer_class(**attn_params)
            
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])#不共享参数
                    
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, feat0, feat1,  q_mask=None, kv_mask=None):
        """
        Args:
            feat0 (torch.Tensor): [B, D, H1, W1]
            feat1 (torch.Tensor): [B, D, H2, W2]
           
        """
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0)
                feat1 = layer(feat1, feat1,q_mask,kv_mask) 
            elif name == 'cross':
                feat0 = layer(feat0, feat1,q_mask=None,kv_mask=kv_mask) 
                feat1 = layer(feat1, feat0,q_mask,kv_mask=None)
            else:
                raise KeyError

        return feat0, feat1


