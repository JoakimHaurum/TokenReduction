import logging
from functools import partial

from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import PatchEmbed, Mlp, DropPath

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }
    
def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


class AdaptiveTokenSampling(nn.Module):
    def __init__(self, sample_count, eps = 1e-6):
        super().__init__()
        self.sample_count = sample_count # K
        self.sample_steps = torch.arange(1/(2*sample_count), (2*sample_count-1)/(2*sample_count), 2/(2*sample_count))

        self.eps = eps

    def forward(self, x, attn, mask):
        cls_attn = attn[:,:, 0, 1:]

        B, H, N = attn.shape[:3]
        # calculate the norms of the values, for weighting the scores, as described in the paper

        value_norms = x[:, :, 1:, :].norm(dim = -1)

        # weigh the attention scores by the norm of the values, sum across all heads
        
        sig_score = cls_attn * value_norms
        sig_score = torch.sum(sig_score, dim=1)
        #cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms)

        # normalize to 1
        normed_sig_score = sig_score / (sig_score.sum(dim = -1, keepdim = True) + self.eps)
        
        cdf = normed_sig_score.cumsum(dim=1)
        cdf[mask[:,1:] == False] += 0.1
        
        # TODO: Maybe not correct? Better to use L1??
        dist = torch.cdist(self.sample_steps.unsqueeze(0).unsqueeze(2).to(cdf.device),cdf.unsqueeze(2))
        sampled_token_ids = torch.argmin(dist, dim=-1) +1 

        
        unique_sampled_token_ids_list = [torch.unique(t, sorted = True) for t in torch.unbind(sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(unique_sampled_token_ids_list, batch_first = True)

        new_mask = unique_sampled_token_ids != 0
        new_mask = F.pad(new_mask, (1, 0), value = True)

        unique_sampled_token_ids = F.pad(unique_sampled_token_ids, (1, 0), value = 0)
        expanded_unique_sampled_token_ids = unique_sampled_token_ids.unsqueeze(1).repeat(1, H, 1)

        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim = 2)

        
        return new_attn, new_mask, unique_sampled_token_ids


class ATSAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., ats_sample_count=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ats_sample_count = ats_sample_count

        if self.ats_sample_count:
            self.ats = AdaptiveTokenSampling(ats_sample_count)

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            dots_mask = mask.unsqueeze(1).unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~dots_mask, mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        sample_ids = None
        if self.ats_sample_count:
            attn, mask, sample_ids = self.ats(v, attn, mask)

        B, _, N, _ = attn.shape
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask, sample_ids


class ATSBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ats_sample_count = 0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ATSAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, ats_sample_count=ats_sample_count)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask):
        x_tmp= self.norm1(x)
        x_tmp, mask, sample_ids = self.attn(x_tmp, mask)

        if sample_ids is not None:
            x = batched_index_select(x, sample_ids, dim = 1)

        x = x + self.drop_path1(x_tmp)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, mask, sample_ids


class ATSVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,  representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, 
                 act_layer=None, weight_init='', args=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
                 drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                 act_layer, weight_init)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.sample_loc = args.reduction_loc
        sample_count = args.keep_rate

        if len(sample_count) == 1:
            sample_count = [int(args.keep_rate[0]**(idx+1) * self.patch_embed.num_patches)+1 for idx in range(len(self.sample_loc))]

        assert len(sample_count) == len(self.sample_loc), f"Mismatch between the sample location ({self.sample_loc}) and sample centers ({sample_count})"
        
        cnt = 0
        self.sample_count = [0]*depth
        for idx in range(depth):
            if idx in self.sample_loc:
                self.sample_count[idx] = sample_count[cnt]
                cnt += 1

        print(self.sample_count, self.sample_loc)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            ATSBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, ats_sample_count=self.sample_count[i])
            for i in range(depth)])

        self.viz_mode = getattr(args, 'viz_mode', False)

        self.apply(self._init_weights)

    def get_new_module_names(self):
        return []
        
    def get_reduction_count(self):
        return self.sample_loc

    def forward(self, x):
        x = self.patch_embed(x)
        B, N = x.shape[:2]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        mask = torch.ones((B, N+self.num_tokens), dtype = torch.bool, device=x.device)
        
        if self.viz_mode:
            decisions = {}
            features = {}

        for idx, blck in enumerate(self.blocks):
            x, mask, sample_ids = blck(x, mask)
            
            if self.viz_mode and sample_ids is not None:
                sample_ids = sample_ids[:, 1:] - 1
                decisions[idx] = sample_ids.clone().detach().cpu().numpy()
                features[idx] = x.clone().detach().cpu().numpy()

        if self.viz_mode and 11 not in features.keys():
            features[idx] = x.clone().detach().cpu().numpy()

        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        if self.training:
            return x
        else:
            if self.viz_mode:
                viz_data = {"Kept_Tokens": decisions, "Features": features}
                return x, viz_data
            else:
                return x