import torch
import torch.nn as nn
from .common import Conv2dBNAct, ResidualConv2dBNAct
from .utils.position_encoding import PositionEncodingSine
from .attention_module.transformer import Multiscale_Linear_attention
from .cross_correlation import BatchDepthwiseCrossCorrelation
from .encoder import Backbone
import math

class P2WNET(nn.Module):
    def __init__(
        self, 
        num_features:int=256, 
        downsampling = 32,
        color_format_search='gray',
        template_size=(512,640),
        search_size=(1536,1920),
        max_shape=(256,256),
        d_model = 256,
        num_features_predition_head=256,
        n_heads = 4,
        layer_names=['self', 'cross'] * 4,
        head_kernel_size=1,
        attention_drop_out_rate=0.0,
        num_of_tasktoken = 4,
        num_of_predited_pts = 4,
        cnn_act = 'gelu',
        att_act = 'gelu',
        cnn_norm: str="batchnorm",
        use_self_attention_before_predition=False,
        cnn_conv2d_has_bias=False,
        cnn_bn_has_bias=True,
        att_scales=(3,5),
        att_conv2d_has_bias=False,
        att_bn_has_bias=True,
        attn_type='MLA',
        size_corr_kernel_size=1,
        use_share_encoder=False,
        cal_gradient = False,
        kernel_fn='relu',
        **kwargs
    ) -> None:    
        super().__init__()
        self.d_model = d_model
        self.search_size = search_size[0]
        self.num_of_tasktoken = num_of_tasktoken
        self.use_self_attention_before_predition = use_self_attention_before_predition
        self.num_of_predited_pts = num_of_predited_pts
        self.size_corr_kernel_size = size_corr_kernel_size
        self.template_h_strided ,self.template_w_strided = template_size[0]//downsampling , template_size[1]//downsampling
        self.pos_encoding = PositionEncodingSine(
                             d_model,
                             max_shape,
                             )
        
        if use_share_encoder:
            self.search_module = Backbone(color_format_search,num_features,d_model,downsampling,0,cnn_act,cnn_norm,cnn_conv2d_has_bias=cnn_conv2d_has_bias,cnn_bn_has_bias=cnn_bn_has_bias,cal_gradient=cal_gradient)
            self.template_module = self.search_module
        else:
            self.search_module =  Backbone(color_format_search,num_features,d_model,downsampling,0,cnn_act,cnn_norm,cnn_conv2d_has_bias=cnn_conv2d_has_bias,cnn_bn_has_bias=cnn_bn_has_bias,cal_gradient=cal_gradient)
            self.template_module =  Backbone(color_format_search,num_features,d_model,downsampling,0,cnn_act,cnn_norm,cnn_conv2d_has_bias=cnn_conv2d_has_bias,cnn_bn_has_bias=cnn_bn_has_bias,cal_gradient=cal_gradient)
        
        
        self.num_of_tasktoken_row = math.ceil(num_of_predited_pts/self.template_w_strided)
        self.task_tokens = nn.Parameter(torch.randn(1,d_model,self.num_of_tasktoken_row,self.template_w_strided))
        self.corr_proj_layer = Conv2dBNAct(d_model,(4*d_model)//num_of_predited_pts,kernel_size=1,stride=1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)
        # applay cross-attention
        self.attention_module = Multiscale_Linear_attention(d_model=d_model,
                                                        nhead=n_heads,
                                                        layer_names=layer_names,
                                                        attention_drop_out_rate=attention_drop_out_rate,
                                                        scales=att_scales,
                                                        att_act=att_act,
                                                        att_conv2d_has_bias=att_conv2d_has_bias,
                                                        att_bn_has_bias=att_bn_has_bias,
                                                        attn_type = attn_type,
                                                        kernel_fn = kernel_fn,
                                                        )

        self.batch_corr = BatchDepthwiseCrossCorrelation()

        in_channels_head = template_size[0]*template_size[1]//(downsampling**2)
        
        out_conv_layers_score = [
            ResidualConv2dBNAct((4*d_model)//num_of_predited_pts*num_of_predited_pts + in_channels_head, num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias),
            ResidualConv2dBNAct(num_features_predition_head , num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias),
            ResidualConv2dBNAct(num_features_predition_head , num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias),
        ]
        self.out_conv = nn.Sequential(*out_conv_layers_score)
        
        self.head_cls = nn.ModuleList([ResidualConv2dBNAct(num_features_predition_head , num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)]+[Conv2dBNAct(num_features_predition_head, num_of_predited_pts, head_kernel_size, 1, has_activation=False, act=cnn_act,norm=None,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)])                  
        self.head_reg = nn.ModuleList([ResidualConv2dBNAct(num_features_predition_head , num_features_predition_head, 3, 1,act=cnn_act,norm=cnn_norm,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)]+[Conv2dBNAct(num_features_predition_head, num_of_predited_pts*2, head_kernel_size, 1, has_activation=False, act=cnn_act,norm=None,conv_has_bias=cnn_conv2d_has_bias,bn_has_bias=cnn_bn_has_bias)])

    def forward(self, imgs_search, imgs_template):
        """
        imgs_search (b,1,search_size, search_size)
        imgs_template (b,1,search_size, search_size)
        output (b,2,search_size)
        """

        x_search, x_search_rs = self.search_module(imgs_search)
        x_template, x_template_rs = self.template_module(imgs_template)

        b,c,h_search,w_search = x_search.shape
        _,_,h_template,w_template = x_template.shape
        
        x_search = self.pos_encoding(x_search)
        x_template = self.pos_encoding(x_template)
        
        tasktoken = self.task_tokens.repeat(b,1,1,1)
        x_template = torch.cat([x_template,tasktoken],dim = 2)
        
        search_out, template_out = self.attention_module(x_search,x_template)
        
        search_out = search_out.reshape(b,-1,h_search*w_search).transpose(-1,-2).contiguous()
        template_out = template_out.reshape(b,-1,(h_template+self.num_of_tasktoken_row)*w_template).transpose(-1,-2).contiguous()

        search_out_copy = search_out.clone()

        tasktoken_out = template_out[:, -self.num_of_tasktoken_row*w_template:, :]
        template_out = template_out[:, :-self.num_of_tasktoken_row*w_template, :]

        search_out, template_out = map(lambda feat: feat / feat.shape[-1]**.5,
                               [search_out, template_out])
       
        sim_matrix = torch.einsum("nlc,nsc->nls",  search_out,template_out) 
   
        sim_matrix_out = sim_matrix.clone().permute(0,-1,-2).reshape(b, h_template*w_template, h_search,w_search).contiguous()

        relation_map=sim_matrix_out.clone()

        search_space = search_out_copy.permute(0,-1,-2).reshape(b, -1, h_search,w_search).contiguous()

        all_templates = tasktoken_out[:, -self.num_of_predited_pts:, :].transpose(0, 1).contiguous()  # [num_pts, b, dim]
        all_templates = all_templates.reshape(self.num_of_predited_pts, b, -1, 1, 1).contiguous()  # [num_pts, b, dim, 1, 1]
        all_templates = all_templates.expand(-1, -1, -1, self.size_corr_kernel_size, self.size_corr_kernel_size)  # [num_pts, b, dim, k, k]

        all_corr_outputs = self.batch_corr(search_space, all_templates)
        num_pts, b, channels, h_s, w_s = all_corr_outputs.shape
        all_corr_outputs_4d = all_corr_outputs.reshape(num_pts * b, channels, h_s, w_s)
        all_corr_outputs_4d = self.corr_proj_layer(all_corr_outputs_4d)
        out_channels = all_corr_outputs_4d.shape[1]
        all_corr_outputs_5d = all_corr_outputs_4d.reshape(num_pts, b, out_channels, h_s, w_s)
        all_corr_outputs_reversed = all_corr_outputs_5d.flip(0)
        all_corr_outputs_concat = all_corr_outputs_reversed.permute(1, 0, 2, 3, 4).reshape(b, -1, h_s, w_s)

        relation_map = torch.cat([all_corr_outputs_concat, relation_map], dim=1)
        relation_map = self.out_conv(relation_map)

        score_map = relation_map
        offset_map = relation_map
        for layer in  self.head_cls:
            score_map = layer(score_map )
        for layer in  self.head_reg:
            offset_map = layer(offset_map )

        return sim_matrix,score_map,offset_map, x_search_rs, x_template_rs

if __name__ =='__main__':
    pass

