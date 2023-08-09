import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import PatchEmbed

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


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
    

def cluster_dpc_knn(x, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
    Args:
        x (Tensor[B,N,C]): Input tokens
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, index_down


def merge_tokens(x, idx_token, agg_weight, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information
    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    return x_merged, idx_token_new, agg_weight_new


class CTM(nn.Module):
    def __init__(self, embed_dim, cluster_num, k=5, equal_weight=False):
        super().__init__()
        self.cluster_num = cluster_num
        self.equal_weight = equal_weight
        self.k = k
        
        if not self.equal_weight:
            self.score = nn.Linear(embed_dim, 1)

    def forward(self, x, idx_token, agg_weight, viz_mode = False):

        if not self.equal_weight:
            token_score = self.score(x)
            token_weight = token_score.exp()
        else:
            token_weight = None

        idx_cluster, idx_centers = cluster_dpc_knn(x, self.cluster_num, self.k)
        
        if viz_mode:
            cluster_centers = index_points(x,idx_centers)

        x, idx_token, agg_weight = merge_tokens(x, idx_token, agg_weight, idx_cluster, self.cluster_num, token_weight)


        if viz_mode:
            return x, idx_token, agg_weight, idx_centers, idx_cluster, cluster_centers            
        else:
            return x, idx_token, agg_weight, None, None, None


class DPCKNNVisionTransformer(VisionTransformer):
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

        self.cluster_loc = args.reduction_loc
        self.cluster_count = args.keep_rate
        self.k_neighbors = args.k_neighbors
        self.equal_weight = args.equal_weight

        if len(self.cluster_count) == 1:
            self.cluster_count = [int(self.patch_embed.num_patches * (args.keep_rate[0] ** (idx+1))) for idx in range(len(self.cluster_loc))]

        assert len(self.cluster_count) == len(self.cluster_loc), f"Mismatch between the cluster location ({self.cluster_loc}) and cluster centers ({self.cluster_count})"
        print(self.cluster_count, self.cluster_loc)

        self.cluster_layers = nn.ModuleList([CTM(embed_dim, self.cluster_count[idx], self.k_neighbors, self.equal_weight) for idx in range(len(self.cluster_loc))])
        self.blocks = nn.ModuleList([*self.blocks])

        self.viz_mode = getattr(args, 'viz_mode', False)

        self.apply(self._init_weights)

    def get_new_module_names(self):
        return ["cluster_layers"]

    def get_reduction_count(self):
        return self.cluster_loc

    def forward(self, x):

        x = self.patch_embed(x)

        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)

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
                x, idx_token, agg_weight, idx_centers, idx_cluster, cluster_centers = self.cluster_layers[cluster_cnt](x[:,self.num_tokens:], idx_token, agg_weight, self.viz_mode)
                x = torch.cat((global_tokens, x), dim=1)
                cluster_cnt += 1
                
                if self.viz_mode:
                    decisions[i] = idx_centers.clone().detach().cpu().numpy()
                    assignments[i] = idx_cluster.clone().detach().cpu().numpy()
                    centers_feats[i] = cluster_centers.clone().detach().cpu().numpy()

            x = blk(x)

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