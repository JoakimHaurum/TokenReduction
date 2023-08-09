import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def k_medoids_fit(x, cluster_num, iterations = 5, token_weight=None):
    B, N, C = x.shape

    if token_weight is None:

        init_cluster_idx = np.random.choice(np.arange(N), 1)

        cluster_idx = torch.ones((B,1), dtype=torch.long, device=x.device)*init_cluster_idx
        
        for k in range(1, cluster_num):
            centers = batched_index_select(x, 1, cluster_idx)
            inter_dist_matrix = torch.cdist(x,centers)

            for idx in range(B):
                for k_tmp in range(k):
                    inter_dist_matrix[idx, cluster_idx[idx, k_tmp]] = 0

            max_dist,_ = torch.max(inter_dist_matrix, dim=-1)
            _, new_cluster_idx = torch.max(max_dist, dim=-1)
            cluster_idx = torch.cat((cluster_idx,new_cluster_idx.reshape(B,-1)),dim=-1)

        token_weight = x.new_ones(B, N, 1)
    else:
        _, cluster_idx = torch.topk(token_weight, k=cluster_num, dim=1)
        cluster_idx = cluster_idx.squeeze(2)
    
    centers = batched_index_select(x, 1, cluster_idx)
    
    dist_matrix = torch.cdist(x, x)
    weighted_dist_matrix = dist_matrix * token_weight

    for _ in range(iterations):
        center_matrix = batched_index_select(dist_matrix, 2, cluster_idx)
        assignment = torch.argmin(center_matrix, dim=-1)

        for k in range(cluster_num):
            weighted_dist_matrix_clone = weighted_dist_matrix.clone()
            weighted_dist_matrix_clone[assignment != k] = 1000000 # Mask out rows not assigned to cluster k
            total_distances = torch.sum(weighted_dist_matrix_clone, dim=-1) # calculate summed distances
            cluster_idx[:,k] = torch.argmin(total_distances, dim=1) # determine point which minimizes distances
                
    center_matrix = batched_index_select(dist_matrix, 2, cluster_idx)
    assignment = torch.argmin(center_matrix, dim=-1)

    centers = batched_index_select(x, 1, cluster_idx)
    return centers, cluster_idx, assignment

    
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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_attn, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

    
class KMedoids(torch.nn.Module):
    def __init__(self, num_clusters, iters, equal_weights = False):
        super().__init__()

        self.cluster_count = num_clusters
        self.iters = iters 
        self.equal_weights = equal_weights

    def forward(self, x, token_weights):
        if self.equal_weights:
            token_weights = None

        x, idx_center, idx_cluster = k_medoids_fit(x, self.cluster_count, self.iters, token_weights)
        return x, idx_center, idx_cluster

    
class KMedoidsVisionTransformer(VisionTransformer):
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

        del(self.blocks)
        self.num_patches = self.patch_embed.num_patches
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cluster_loc = args.reduction_loc
        self.cluster_count = args.keep_rate
        self.cluster_iters = args.cluster_iters
        self.equal_weight = args.equal_weight

        if len(self.cluster_count) == 1:
            self.cluster_count = [int(self.patch_embed.num_patches * (args.keep_rate[0] ** (idx+1))) for idx in range(len(self.cluster_loc))]

        assert len(self.cluster_count) == len(self.cluster_loc), f"Mismatch between the cluster location ({self.cluster_loc}) and cluster centers ({self.cluster_count})"
        print(self.cluster_count, self.cluster_loc)

        self.cluster_layers = nn.ModuleList([KMedoids(self.cluster_count[idx], self.cluster_iters, self.equal_weight) for idx in range(len(self.cluster_loc))])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.viz_mode = getattr(args, 'viz_mode', False)

        self.apply(self._init_weights)

    def get_new_module_names(self):
        return ["cluster_layers"]

    def get_reduction_count(self):
        return self.cluster_loc

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        cluster_cnt = 0
        
        if self.viz_mode:
            decisions = {}
            assignments = {}
            centers_feats = {}
            features = {}

        for i, blk in enumerate(self.blocks):
            if i in self.cluster_loc:
                global_tokens = x[:,:self.num_tokens]
                token_weights = torch.sum(torch.sum(attn, dim=1), dim = 1)[:, self.num_tokens:].unsqueeze(2)
                x, idx_centers, idx_cluster = self.cluster_layers[cluster_cnt](x[:,self.num_tokens:], token_weights)  

                if self.viz_mode:
                    decisions[i] = idx_centers.clone().detach().cpu().numpy()
                    assignments[i] = idx_cluster.clone().detach().cpu().numpy()
                    centers_feats[i] = x.clone().detach().cpu().numpy()

                x = torch.cat((global_tokens, x), dim=1)

                cluster_cnt += 1
            x, attn = blk(x)
            
            if self.viz_mode:
                features[i] = x.clone().detach().cpu().numpy()

        if self.viz_mode and 11 not in features.keys():
            features[i] = x.clone().detach().cpu().numpy()
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        if self.training:
            return x
        else:
            if self.viz_mode:
                viz_data = {"Kept_Tokens": decisions, "Assignment_Maps": assignments, "Center_Feats": centers_feats, "Features": features}
                return x, viz_data
            else:
                return x