import logging
from functools import partial

import numpy as np
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
        

class MaskedHeuristicAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


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

        B, _, N, _ = attn.shape
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HeuristicBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedHeuristicAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask):
        x_tmp= self.norm1(x)
        x = x + self.drop_path1(self.attn(x_tmp, mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, mask


class HeuristicVisionTransformer(VisionTransformer):
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

        self.heuristic_pattern = args.heuristic_pattern
        self.depth = depth
        if args.not_contiguous:
            self.reduction_loc = args.reduction_loc
            self.keep_rate = args.keep_rate

            if len(self.keep_rate) == 1:
                num_tokens = [int(self.patch_embed.num_patches * self.keep_rate[0]**(idx+1)) for idx in range(len(self.reduction_loc))]
            
            assert len(num_tokens) == len(self.reduction_loc), f"Mismatch between the pruning location ({self.reduction_loc}) and token ratios ({num_tokens})"
            print(num_tokens)
            
            self.distances, self.threshold, self.P = self.prep_pattern_stage_subset(num_tokens = num_tokens)
        else:
            self.heuristic_pattern = args.heuristic_pattern
            self.min_radius = args.min_radius
            self.start_stage = int(min(args.reduction_loc))
            self.end_stage = int(max(args.reduction_loc))

            self.reduction_loc = [idx for idx in range(self.start_stage, self.end_stage+1)]

            self.distances, self.threshold, self.P = self.prep_pattern()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            HeuristicBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.viz_mode = getattr(args, 'viz_mode', False)

        self.apply(self._init_weights)
            
    def prep_pattern(self):
        P = int(self.patch_embed.num_patches ** 0.5)

        xs = torch.linspace(-P//2, P//2, steps=P)
        ys = torch.linspace(-P//2, P//2, steps=P)
        x, y = torch.meshgrid(xs, ys)

        if self.heuristic_pattern.lower() == "l1":
            z = torch.abs(x)+torch.abs(y)
        elif self.heuristic_pattern.lower() == "l2":
            z = torch.sqrt(x * x + y * y)
        elif self.heuristic_pattern.lower() == "linf":
            z = torch.max(torch.abs(x), torch.abs(y))

        if self.min_radius is None or self.min_radius <= 0:
            self.min_radius = z[P//2, P//2]

        steps = self.end_stage-self.start_stage+3 # plus 3 to have start and end radiuses not be in the reduction range

        threshold = torch.linspace(z[0,0], self.min_radius, steps)
        threshold = F.pad(threshold, (max(self.start_stage-1,0),0), value = z[0,0])
        threshold = F.pad(threshold, (0,max(self.depth-self.end_stage-1,0)), value = threshold[-1])
        return z, threshold, P


    def prep_pattern_stage_subset(self, num_tokens):
        P = int(self.patch_embed.num_patches ** 0.5)
        xs = torch.linspace(-P//2, P//2, steps=P)
        ys = torch.linspace(-P//2, P//2, steps=P)
        x, y = torch.meshgrid(xs, ys)

        if self.heuristic_pattern.lower() == "l1":
            z = torch.abs(x)+torch.abs(y)
        elif self.heuristic_pattern.lower() == "l2":
            z = torch.sqrt(x * x + y * y)
        elif self.heuristic_pattern.lower() == "linf":
            z = torch.max(torch.abs(x), torch.abs(y))

        unique_distances = torch.unique(z)
        tokens_within_distance = []
        for idx in range(len(unique_distances)):
            tokens_within_distance.append(torch.sum(z <= unique_distances[idx]).item())

        closest_thresholds = []
        for num_token in num_tokens:
            closest_tokens = np.inf
            threshold_distance = None
            threshold_num_tokens = None
            for idx, thresholded_tokens in enumerate(tokens_within_distance):
                if np.abs(num_token - thresholded_tokens) < closest_tokens:
                    closest_tokens = np.abs(num_token - thresholded_tokens)
                    threshold_distance = unique_distances[idx].item()
                    threshold_num_tokens = thresholded_tokens
            
            closest_thresholds.append(threshold_distance)  
            print(threshold_num_tokens)
        closest_thresholds = [unique_distances[-1].item()] + closest_thresholds

        threshold = []
        threshold_counter = 0
        for idx in range(self.depth):
            if idx in self.reduction_loc:
                threshold_counter += 1
            threshold.append(torch.ones((P, P))*closest_thresholds[threshold_counter])

        return z, threshold, P


    def get_new_module_names(self):
        return []

    def get_reduction_count(self):
        return self.reduction_loc

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        
        if self.viz_mode:
            decisions = {}
            features = {}
            
        mask = None
        for idx, blck in enumerate(self.blocks):

            if idx in self.reduction_loc:
                mask = self.distances <= self.threshold[idx]
                mask = mask.reshape(self.P*self.P)

                if self.viz_mode:
                    indices = mask.nonzero(as_tuple=True)[0]
                    indices = indices.unsqueeze(0).expand(B, -1)
                    decisions[idx] = indices.clone().detach().cpu().numpy()

                mask = F.pad(mask, (self.num_tokens,0), value = True)
                mask = mask.unsqueeze(0).to(x.device)
            x, mask = blck(x, mask)

            if idx in self.reduction_loc and self.viz_mode:
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
                viz_data = {"Kept_Tokens_Abs": decisions, "Features": features}
                return x, viz_data
            else:
                return x