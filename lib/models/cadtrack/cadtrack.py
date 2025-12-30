import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.cadtrack.vit_care import vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from timm.models.layers import Mlp
from lib.models.layers.CDA_Module import CDA
from lib.models.layers.attn import Attention_qkv
from functools import partial
import torch.nn.functional as F

class CADTrack(nn.Module):

    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER"):
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.box_head = box_head
        self.offset_range_factor = cfg.MODEL.CDA.OFFSET
        self.track_query_len = cfg.MODEL.CDA.TRACK_QUERY
        self.template_number = cfg.DATA.TEMPLATE.NUMBER

        self.CDA = CDA(q_size=(8, 8), window_size=(8, 8), ksize=4,
                       stride=2,
                       stride_block=(8, 8),
                       offset_range_factor=self.offset_range_factor,
                       share=False)
        self.CSS_strengthen_r = Attention_qkv(hidden_dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.CSS_process_r = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4.), act_layer=nn.GELU,
                                 drop=0.)
        self.CSS_strengthen_x = Attention_qkv(hidden_dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.CSS_process_x = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4.), act_layer=nn.GELU,
                                 drop=0.)
        self.decode_fuse_search = conv(hidden_dim*2, hidden_dim)
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                track_query_before = None
                ):

        out_dict = []
        for i in range(len(search)):
            x, aux_dict, len_zx = self.backbone(z=template, x=search[i], track_query_before=track_query_before
                                                )
            num_template_token = len_zx[0]
            num_search_token = len_zx[1]

            B, N, _ = x.size()
            temp_r = x[:, :N // 2, :]
            temp_x = x[:, N // 2:, :]

            boss_fea = torch.cat([temp_r[:, :1, :], temp_x[:, :1, :]], dim=1)
            temp_r_str, temp_x_str = self.CDA(temp_r[:, 1:num_template_token // 2 + 1, :],
                                              temp_r[:, num_template_token // 2 + 1:num_template_token + 1, :],
                                              temp_x[:, 1:num_template_token // 2 + 1, :],
                                              temp_x[:, num_template_token // 2 + 1:num_template_token + 1, :],
                                              boss_fea)
            temp_r_query = temp_r_str.clone().detach()
            temp_x_query = temp_x_str.clone().detach()
            track_query_before = [temp_r_query, temp_x_query]

            feat_last_r = temp_r[:, -num_search_token:, :]
            temp_attn_r = temp_r_str + self.CSS_strengthen_r(temp_r_str, feat_last_r, feat_last_r)
            temp_attn_r = temp_attn_r + self.CSS_process_r(temp_attn_r)
            att_r = torch.matmul(feat_last_r, temp_attn_r.transpose(1, 2))
            feat_last_r = att_r * feat_last_r

            feat_last_x = temp_x[:, -num_search_token:, :]
            temp_attn_x = temp_x_str + self.CSS_strengthen_x(temp_x_str, feat_last_x, feat_last_x)
            temp_attn_x = temp_attn_x + self.CSS_process_x(temp_attn_x)
            att_x = torch.matmul(feat_last_x, temp_attn_x.transpose(1, 2))
            feat_last_x = att_x * feat_last_x

            feat_last = torch.cat([feat_last_r, feat_last_x], dim=-1)

            out = self.forward_head(feat_last, None)

            out.update(aux_dict)
            out['track_query_before'] = track_query_before
            out['backbone_feat'] = feat_last
            out_dict.append(out)
        return out_dict

    def forward_head(self, cat_feature ,gt_score_map=None):
        opt = (cat_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.decode_fuse_search(opt_feat)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_cadtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('CADTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            cross_loc=cfg.MODEL.BACKBONE.CROSS_LOC,
                                            )
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = CADTrack(
        backbone,
        box_head,
        cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if training and (
            'OSTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE):
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict_rgbt = dict()
        if 'DropTrack' in cfg.MODEL.PRETRAIN_FILE:
            for k, v in checkpoint["net"].items():
                if k in ['box_head.conv1_ctr.0.weight', 'box_head.conv1_offset.0.weight',
                         'box_head.conv1_size.0.weight']:
                    v = v
                elif 'pos_embed_x' in k:
                    v = resize_pos_embed(v, 16, 16) + checkpoint["net"]['backbone.temporal_pos_embed_x']
                elif 'pos_embed_z' in k:
                    v = resize_pos_embed(v, 8, 8) + checkpoint["net"]['backbone.temporal_pos_embed_z']
                else:
                    v = v
                param_dict_rgbt[k] = v

        else:
            param_dict_rgbt = checkpoint["net"]

        missing_keys, unexpected_keys = model.load_state_dict(param_dict_rgbt, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys', missing_keys, unexpected_keys)
    return model

def resize_pos_embed(posemb, hight, width):
    posemb_grid = posemb[0, :]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print(
        'Resized position embedding from size:{} to new token with height:{} width: {}'.format(posemb_grid.shape, hight,
                                                                                               width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    return posemb_grid
