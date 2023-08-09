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
  

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)

    # log_mu -> log(a) 
    # log_nu -> log(b) 

    # K = exp(C/eps)

    # u = a / Kv -> log(a/Kv) -> log_mu - log(Kv) -> log_mu - log(K) + log(v)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, eps, iters):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    norm = - (ms + ns).log() #-M+N in log space
    log_mu = norm.expand(m) # 1xM vector with value -log(M+N)
    log_nu = norm.expand(n) # 1xN vector with value -log(M+N)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1) # add batch dimension

    # scores is the expoenential 
    Z = log_sinkhorn_iterations(scores/eps, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    Z = Z.exp()
    return Z

    
class Sinkhorn(torch.nn.Module):
    def __init__(self, embed_dim, cluster_centers, eps, iters):
        super().__init__()
        self.v = nn.Parameter(torch.randn(cluster_centers, embed_dim)) # Cluster centers, drawn from zero mean unit variance Gaussian
        self.eps = eps 
        self.iters = iters 

    def forward(self, x):

        b, _, d = x.shape

        x = F.normalize(x, p = 2, dim = -1)
        
        ## Ensure clusters are always unit vectors, and require no gradients
        with torch.no_grad():
            w = self.v.clone()
            w = F.normalize(w, p=2, dim = -1)
            self.v.copy_(w)
        clusters  = self.v[None].expand(b, -1, d)# B x Cluster x D'
        
        scores = torch.bmm(x, clusters.transpose(1,2)) # B x HW x D' and B  x D' x Cluster  -> B x HW x Cluster

        weights = log_optimal_transport(scores.transpose(1,2), self.eps, self.iters).transpose(1,2)

        # Batch matrix multiply approach:  B x D x HW and B x HW x Clusters  =  B x D x Cluseters   -> B x Clusters x D
        x = torch.bmm(x.transpose(1,2), weights).transpose(1,2) # Per input example calculate the weighted sum across the features for each cluster k       
        
        return x, weights.transpose(1,2)


class SinkhornVisionTransformer(VisionTransformer):
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
        self.sinkhorn_eps = args.sinkhorn_eps
        self.sinkhorn_iters = args.cluster_iters

        if len(self.cluster_count) == 1:
            self.cluster_count = [int(self.patch_embed.num_patches * (args.keep_rate[0] ** (idx+1))) for idx in range(len(self.cluster_loc))]

        assert len(self.cluster_count) == len(self.cluster_loc), f"Mismatch between the cluster location ({self.cluster_loc}) and cluster centers ({self.cluster_count})"
        print(self.cluster_count, self.cluster_loc)

        self.cluster_layers = nn.ModuleList([Sinkhorn(embed_dim, self.cluster_count[idx], self.sinkhorn_eps, self.sinkhorn_iters) for idx in range(len(self.cluster_loc))])
        self.blocks = nn.ModuleList([*self.blocks])

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
            assignments = {}
            hard_assignment = {}
            centers_feats = {}
            features = {}

        for i, blk in enumerate(self.blocks):
            if i in self.cluster_loc:
                global_tokens = x[:,:self.num_tokens]
                x, soft_assign = self.cluster_layers[cluster_cnt](x[:,self.num_tokens:])
                x = torch.cat((global_tokens, x), dim=1)

                if self.viz_mode:
                    hard_assign = torch.argmax(soft_assign, dim=-2)

                    assignments[i] = soft_assign.clone().detach().cpu().numpy()
                    hard_assignment[i] = hard_assign.clone().detach().cpu().numpy()
                    
                    centers_feat = self.cluster_layers[cluster_cnt].v
                                        
                    centers_feat = centers_feat.unsqueeze(0).expand(B, -1, -1)
                    centers_feats[i] = centers_feat.clone().detach().cpu().numpy()
                cluster_cnt += 1
                    
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
                viz_data = {"Assignment_Maps": hard_assignment, "Soft_Assignment_Maps": assignments, "Center_Feats": centers_feats, "Features": features}
                return x, viz_data
            else:
                return x