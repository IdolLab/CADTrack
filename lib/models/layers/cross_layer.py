from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd

def candidate_elimination(tokens: torch.Tensor, attn: torch.Tensor, template_mask: torch.Tensor, lens_q: int, lens_t: int, keep_ratio: float):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        lens_q (int): length of spatio token
        keep_ratio (float): keep ratio of search region tokens (candidates)
    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
    """
    
    lens_vision = attn.shape[-1] - lens_q
    bs, hn, _, _ = attn.shape
    lens_s = lens_vision - lens_t
    lens_keep = math.ceil(keep_ratio * (lens_vision - lens_t))


    

    tokens_t = tokens[:,:lens_t,:]
    tokens_s = tokens[:,lens_t:lens_vision,:]
    tokens_q = tokens[:,-lens_q:,:]

    attn_t = attn[:, :, :lens_t, lens_t:lens_vision]
    if template_mask is not None:
        template_mask = template_mask.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, template_mask, :]
        attn_t = attn_t[template_mask]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    attn_q = attn[:, :, -lens_q:, lens_t:lens_vision]
    attn_q = attn_q.mean(dim=2).mean(dim=1)

    attn_vision = attn_q + attn_t

    token_mask = torch.ones_like(tokens_s).to('cuda')
    zeros_mask = torch.zeros_like(tokens_s).to('cuda')
    global_index = torch.arange(lens_vision).expand(bs, -1).to('cuda')

    # attn_vision_t = attn[:, :, :lens_t, :lens_vision]
    # attn_vision = torch.cat([attn_vision_q,attn_vision_t],dim=2)
      # B, H, L-T, L_s --> B, L_s

    sorted_attn, indices = torch.sort(attn_vision, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    
    
    token_mask.scatter_(dim=1, index=removed_index.unsqueeze(-1).expand(-1, -1, tokens.size(-1)), src=zeros_mask)
    # print(token_mask)
    tokens_s_new = tokens_s * token_mask

    tokens_new = torch.cat([tokens_t, tokens_s_new, tokens_q],dim=1)
    # for b in range(bs):
    #     for idx in removed_index[b]:
    #         print(torch.all(tokens_new[b, idx, :] == 1))

    return tokens_new



class Up_Down(nn.Module):
    def __init__(self, dim, xavier_init=False):
        super().__init__()
 
        self.adapter_down = nn.Linear(dim, dim//4)  
        self.adapter_up = nn.Linear(dim//4, dim//2)  
        self.adapter_mid = nn.Linear(dim//4, dim//4)
        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = nn.GELU()
        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        # x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        x_down = self.act(x_down)
        # x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up