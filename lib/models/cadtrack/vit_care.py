import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.cadtrack.base_backbone import BaseBackbone
from lib.models.cadtrack.utils import combine_tokens, recover_tokens, generate_template_mask
from lib.models.layers.cross_layer import *
from lib.models.layers.adapter import Mamba_adapter

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xi):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        xi = xi + self.drop_path(self.attn(self.norm1(xi)))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        return x, xi

class MCILayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adap_decline_x = nn.Linear(dim, 8)
        self.adap_decline_xi = nn.Linear(dim, 8)
        self.adap_share = Mamba_adapter(dim=8)
        self.adap_incline_x = nn.Linear(8, dim)
        self.adap_incline_xi = nn.Linear(8, dim)

    def forward(self, x, xi):
        x_adap = self.adap_decline_x(x)
        xi_adap = self.adap_decline_xi(xi)
        cat_adap = self.adap_share(x_adap, xi_adap)
        x_adap, xi_adap = torch.split(cat_adap, cat_adap.shape[1] // 2, dim=1)
        x_adap = self.adap_incline_x(x_adap)
        xi_adap = self.adap_incline_xi(xi_adap)
        x = x + x_adap
        xi = xi + xi_adap

        return x, xi

class VisionTransformer(BaseBackbone):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 cross_loc=None, drop_path=None, add_cls_token=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.add_cls_token = add_cls_token

        self.cross_loc = cross_loc
        self.MCI = nn.ModuleList()
        if self.cross_loc is not None and type(self.cross_loc) == list:
            for i in range(len(self.cross_loc)):
                self.MCI.append(
                    MCILayer(dim=embed_dim))

        self.moe_proj_layers = nn.Sequential(*[
            nn.Linear(embed_dim, embed_dim)
            for i in range(depth-1)])
        self.moe_avg = nn.AdaptiveAvgPool2d((768, 1280))
        self.moe_mlp = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        self.moe_fc = nn.Linear(768 * 128, 10)
        self.moe_proj_weights = nn.Parameter(torch.ones(1, 6))

        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def forward_features(self, z, x, track_query_before):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        number = len(z)

        x_r_list = []
        x_x_list = []

        for i in range(number):
            x_r_list.append(self.patch_embed(z[i][:, :3]))
            x_x_list.append(self.patch_embed(z[i][:, 3:]))
            x_r_list[i] += self.pos_embed_z
            x_x_list[i] += self.pos_embed_z

        x_r = self.patch_embed(x[:, :3])
        x_x = self.patch_embed(x[:, 3:])
        x_r += self.pos_embed_x
        x_x += self.pos_embed_x

        x_r_list.append(x_r)
        x_x_list.append(x_x)

        x_r = torch.cat(x_r_list, dim=1)
        x_x = torch.cat(x_x_list, dim=1)

        x_r = self.pos_drop(x_r)
        x_x = self.pos_drop(x_x)
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        lens_z = lens_z * number

        new_query = self.cls_token.expand(B, 1, -1)  # copy B times
        track_query_r = new_query if track_query_before is None else track_query_before[0] + new_query
        track_query_x = new_query if track_query_before is None else track_query_before[1] + new_query
        x_r = torch.cat([track_query_r, x_r], dim=1)
        x_x = torch.cat([track_query_x, x_x], dim=1)

        cross_index = 0
        res_x_r = []
        res_x_x = []
        res_proj_x_r = []
        res_proj_x_x = []
        for i, blk in enumerate(self.blocks):
            x_r, x_x = blk(x_r, x_x)

            if self.cross_loc is not None and i in self.cross_loc:
                x_r, x_x = self.MCI[cross_index](x_r, x_x)
                cross_index += 1

            if i < self.depth-1:
                proj_x_r = self.moe_proj_layers[i](x_r)
                proj_x_x = self.moe_proj_layers[i](x_x)
            else:
                proj_x_r = x_r
                proj_x_x = x_x
            res_x_r.append(x_r[:, lens_z+1:, :])
            res_x_x.append(x_x[:, lens_z+1:, :])
            res_proj_x_r.append(proj_x_r[:, lens_z+1:, :])
            res_proj_x_x.append(proj_x_x[:, lens_z+1:, :])

        shift_indice_x_r, selected_proj_x_r = self.shift(res_x_r, res_proj_x_r)
        selected_proj_x_r = selected_proj_x_r.sum(dim=1)
        x_r = torch.cat([x_r[:, :lens_z+1, :], selected_proj_x_r], dim=1)

        shift_indice_x_x, selected_proj_x_x = self.shift(res_x_x, res_proj_x_x)
        selected_proj_x_x = selected_proj_x_x.sum(dim=1)
        x_x = torch.cat([x_x[:, :lens_z+1, :], selected_proj_x_x], dim=1)

        x_r = recover_tokens(x_r, lens_z, lens_x, mode=self.cat_mode)
        x_x = recover_tokens(x_x, lens_z, lens_x, mode=self.cat_mode)

        x = torch.cat([x_r, x_x], dim=1)
        len_zx = [lens_z, lens_x]
        aux_dict = {"attn": None,
        }
        return self.norm(x), aux_dict, len_zx

    def shift(self, guide, x):
        x_tensor = torch.stack(x, dim=1)
        B, _, L, C = x_tensor.shape

        guide = torch.cat(guide[1:-1], 1)
        guide = guide.permute(0, 2, 1)
        x_avg = self.moe_avg(guide)  # B, C, 1280
        logit = self.moe_mlp(x_avg)  # B, C, 128
        logit = self.moe_fc(logit.reshape(B, -1))
        _, shift_indice = torch.sort(logit, dim=1)
        shift_indice = shift_indice + 1

        zeros = torch.zeros((B, 1), dtype=shift_indice.dtype, device=shift_indice.device)
        elevens = torch.full((B, 1), 11, dtype=shift_indice.dtype, device=shift_indice.device)
        top_4 = shift_indice[:, :4]
        combined = torch.cat([zeros, top_4, elevens], dim=1)
        sorted_combined, _ = torch.sort(combined, dim=1)
        sorted_combined_expanded = sorted_combined.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, C)
        selected_layers = torch.gather(x_tensor, 1, sorted_combined_expanded)
        proj_weights = self.moe_proj_weights.expand(B, -1)
        selected_proj = selected_layers * proj_weights.unsqueeze(-1).unsqueeze(-1)

        return sorted_combined, selected_proj


    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_xmpl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_xmpl: bool = False):

    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_xmpl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_xmpl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):

    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=False, **model_kwargs)
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Small model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Tiny model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model