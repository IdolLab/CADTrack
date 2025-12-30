import torch.nn as nn
import einops
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from timm.models.layers import trunc_normal_
import textwrap


class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, ksize, share
    ):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.stride = stride
        kk = self.ksize
        pad_size = 0
        self.share_offset = share
        if self.share_offset:
            self.conv_offset = nn.Sequential(
                nn.Conv2d(4 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            )
        else:
            self.conv_offset_x_r1 = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_x_r2 = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_x_x1 = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_x_x2 = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _get_ref_points(self, H_in, W_in, B, kernel_size, stride, dtype, device):

        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

        center_y = torch.arange(H_out, dtype=dtype, device=device) * stride + (kernel_size // 2)
        center_x = torch.arange(W_out, dtype=dtype, device=device) * stride + (kernel_size // 2)

        ref_y, ref_x = torch.meshgrid(center_y, center_x, indexing='ij')
        ref = torch.stack((ref_y, ref_x), dim=-1)

        ref[..., 1].div_(W_in - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_in - 1.0).mul_(2.0).sub_(1.0)

        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)

        return ref

    def off_set_shared(self, data, reference):
        data = einops.rearrange(data, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=4 * self.n_group_channels)
        offset = self.conv_offset(data)
        Hk, Wk = offset.size(2), offset.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        pos_x_r1 = (offset + reference).clamp(-1., +1.)
        pos_x_r2 = (offset + reference).clamp(-1., +1.)
        pos_x_x1 = (offset + reference).clamp(-1., +1.)
        pos_x_x2 = (offset + reference).clamp(-1., +1.)
        return pos_x_r1, pos_x_r2, pos_x_x1, pos_x_x2, Hk, Wk

    def off_set_unshared(self, data, reference):
        x_r1, x_r2, x_x1, x_x2 = data.chunk(4, dim=1)
        x_r1 = einops.rearrange(x_r1, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        x_r2 = einops.rearrange(x_r2, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        x_x1 = einops.rearrange(x_x1, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        x_x2 = einops.rearrange(x_x2, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset_x_r1 = self.conv_offset_x_r1(x_r1)
        offset_x_r2 = self.conv_offset_x_r2(x_r2)
        offset_x_x1 = self.conv_offset_x_x1(x_x1)
        offset_x_x2 = self.conv_offset_x_x2(x_x2)
        Hk, Wk = offset_x_r1.size(2), offset_x_r1.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset_x_r1 = offset_x_r1.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_x_r2 = offset_x_r2.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_x_x1 = offset_x_x1.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_x_x2 = offset_x_x2.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset_x_r1 = einops.rearrange(offset_x_r1, 'b p h w -> b h w p')
        offset_x_r2 = einops.rearrange(offset_x_r2, 'b p h w -> b h w p')
        offset_x_x1 = einops.rearrange(offset_x_x1, 'b p h w -> b h w p')
        offset_x_x2 = einops.rearrange(offset_x_x2, 'b p h w -> b h w p')
        pos_x_r1 = (offset_x_r1 + reference).clamp(-1., +1.)
        pos_x_r2 = (offset_x_r2 + reference).clamp(-1., +1.)
        pos_x_x1 = (offset_x_x1 + reference).clamp(-1., +1.)
        pos_x_x2 = (offset_x_x2 + reference).clamp(-1., +1.)
        return pos_x_r1, pos_x_r2, pos_x_x1, pos_x_x2, Hk, Wk

    def forward(self, query, x_r1, x_r2, x_x1, x_x2, writer=None, epoch=None, img_path=None, text=''):
        B, C, H, W = x_r1.size()
        b_, c_, h_, w_ = query.size()
        dtype, device = x_r1.dtype, x_r1.device
        data = torch.cat([x_r1, x_r2, x_x1, x_x2], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)
        if self.share_offset:
            pos_x_r1, pos_x_r2, pos_x_x1, pos_x_x2, Hk, Wk = self.off_set_shared(data, reference)
        else:
            pos_x_r1, pos_x_r2, pos_x_x1, pos_x_x2, Hk, Wk = self.off_set_unshared(data, reference)
        n_sample = Hk * Wk
        sampled_x_r1 = F.grid_sample(
            input=x_r1.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x_r1[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        sampled_x_r2 = F.grid_sample(
            input=x_r2.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x_r2[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_x_x1 = F.grid_sample(
            input=x_x1.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x_x1[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_x_x2 = F.grid_sample(
            input=x_x2.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x_x2[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)

        sampled_x_r1 = sampled_x_r1.reshape(B, C, 1, n_sample)
        sampled_x_r2 = sampled_x_r2.reshape(B, C, 1, n_sample)
        sampled_x_x1 = sampled_x_x1.reshape(B, C, 1, n_sample)
        sampled_x_x2 = sampled_x_x2.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x_r1, sampled_x_r2, sampled_x_x1, sampled_x_x2], dim=-1)

        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 4 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 4 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)

        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        if self.share_offset:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_shared(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)
        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * Hk * Wk)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woSample_wCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woSample_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * n_sample)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woOffset(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        data = torch.cat([x, y, z], dim=1)
        x = self.conv_v(data)
        y = self.conv_n(data)
        z = self.conv_t(data)
        h_new, w_new = x.size(2), x.size(3)
        n_sample = h_new * w_new
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)


class CDA(nn.Module):

    def __init__(self, window_size=(5, 5), q_size=(8, 8), n_heads=1, n_head_channels=768, n_groups=1, attn_drop=0.,
                 proj_drop=0., stride=2, stride_block=(4, 4),
                 offset_range_factor=5, ksize=5, share=False):
        super(CDA, self).__init__()
        self.q_size = q_size
        self.window_size = window_size
        self.stride_block = stride_block
        self.feat_dim = n_head_channels * n_heads
        self.num_da = self.calculate_num_blocks(q_size, window_size, stride_block)
        self.da_group = nn.ModuleList([
            DAttentionBaseline(
                window_size, n_heads, n_head_channels, n_groups, attn_drop, proj_drop, stride,
                offset_range_factor, ksize, share
            ) for _ in range(self.num_da)
        ])

    def calculate_num_blocks(self, input_size, block_size, stride):
        H, W = input_size
        block_h, block_w = block_size
        stride_h, stride_w = stride

        num_blocks_h = (H - block_h) // stride_h + 1
        num_blocks_w = (W - block_w) // stride_w + 1

        return num_blocks_h * num_blocks_w

    def split_into_blocks_with_overlap(self, input_tensor, block_size=(4, 4), stride=(4, 4)):

        B, C, H, W = input_tensor.shape
        block_h, block_w = block_size
        stride_h, stride_w = stride

        assert H >= block_h and W >= block_w, "Block size should be smaller than the input feature map."

        unfolded = input_tensor.unfold(2, block_h, stride_h).unfold(3, block_w, stride_w)

        unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()

        return unfolded

    def forward(self, x_r1, x_r2, x_x1, x_x2, boss, writer=None, epoch=None, img_path=None, texts=''):
        x_r1 = x_r1.reshape(x_r1.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_r2 = x_r2.reshape(x_r2.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_x1 = x_x1.reshape(x_x1.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_x2 = x_x2.reshape(x_x2.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_r1_blocks = self.split_into_blocks_with_overlap(x_r1, self.window_size, self.stride_block).flatten(1, 2)
        x_r2_blocks = self.split_into_blocks_with_overlap(x_r2, self.window_size, self.stride_block).flatten(1, 2)
        x_x1_blocks = self.split_into_blocks_with_overlap(x_x1, self.window_size, self.stride_block).flatten(1, 2)
        x_x2_blocks = self.split_into_blocks_with_overlap(x_x2, self.window_size, self.stride_block).flatten(1, 2)
        boss = boss.permute(0, 2, 1).unsqueeze(-2)
        query_cash = []
        for i in range(self.num_da):
            query_cash.append(
                self.da_group[i](boss, x_r1_blocks[:, i], x_r2_blocks[:, i], x_x1_blocks[:, i], x_x2_blocks[:, i], writer=writer, epoch=epoch,
                                 img_path=img_path, text=texts).squeeze(-1))
        fea = query_cash[0].permute(0, 2, 1)
        track_x, track_r = torch.split(fea, fea.size(1) // 2, dim=1)

        return track_x, track_r
