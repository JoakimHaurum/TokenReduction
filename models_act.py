import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import _cfg, default_cfgs
from timm.models.registry import register_model

__all__ = [
    'deit_tiny_patch16_224_local', \
    'deit_small_patch16_224_local', \
    'deit_base_patch16_224_local', \
    'deit_tiny_patch16_224_local_viz', \
    'deit_small_patch16_224_local_viz', \
    'deit_base_patch16_224_local_viz', \
    "dpcknn_tiny_patch16_224", \
    "dpcknn_small_patch16_224", \
    "dpcknn_base_patch16_224", \
    "dyvit_tiny_patch16_224", \
    "dyvit_small_patch16_224", \
    "dyvit_base_patch16_224", \
    "dyvit_tiny_patch16_224_teacher", \
    "dyvit_small_patch16_224_teacher", \
    "dyvit_base_patch16_224_teacher", \
    "kmedoids_tiny_patch16_224", \
    "kmedoids_small_patch16_224", \
    "kmedoids_base_patch16_224", \
    "patchmerger_tiny_patch16_224", \
    "patchmerger_small_patch16_224", \
    "patchmerger_base_patch16_224", \
    "sinkhorn_tiny_patch16_224", \
    "sinkhorn_small_patch16_224", \
    "sinkhorn_base_patch16_224", \
    "ats_tiny_patch16_224", \
    "ats_small_patch16_224", \
    "ats_base_patch16_224", \
    "heuristic_tiny_patch16_224", \
    "heuristic_small_patch16_224", \
    "heuristic_base_patch16_224", \
    "topk_tiny_patch16_224", \
    "topk_small_patch16_224", \
    "topk_base_patch16_224",  \
    "evit_tiny_patch16_224", \
    "evit_small_patch16_224", \
    "evit_base_patch16_224",  \
    "tome_tiny_patch16_224", \
    "tome_small_patch16_224", \
    "tome_base_patch16_224", \
    "sit_tiny_patch16_224", \
    "sit_small_patch16_224", \
    "sit_base_patch16_224", \
]


deit_url_paths = {"deit_tiny_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                  "deit_tiny_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                  "deit_small_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                  "deit_small_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
                  "deit_base_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                  "deit_base_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
                  }


@register_model
def deit_tiny_patch16_224_local(pretrained=False, **kwargs):

    from timm.models.vision_transformer import VisionTransformer

    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
        
    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]
    if pretrained:

        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )

        print(checkpoint["model"].keys())
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def deit_small_patch16_224_local(pretrained=False, **kwargs):

    from timm.models.vision_transformer import VisionTransformer

    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_base_patch16_224_local(pretrained=False, **kwargs):

    from timm.models.vision_transformer import VisionTransformer
    
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


    
@register_model
def deit_tiny_patch16_224_local_viz(pretrained=False, **kwargs):

    from models.deit_viz import VisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
        
    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]
    if pretrained:

        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )

        print(checkpoint["model"].keys())
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def deit_small_patch16_224_local_viz(pretrained=False, **kwargs):

    from models.deit_viz import VisionTransformer

    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_base_patch16_224_local_viz(pretrained=False, **kwargs):

    from models.deit_viz import VisionTransformer
    
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def dyvit_tiny_patch16_224(pretrained=False, **kwargs):

    from models.dyvit import DynamicVisionTransformer

    dyvit_distillation = kwargs["args"].dyvit_distill
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DynamicVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        dyvit_distillation=dyvit_distillation, distilled=deit_distillation, **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model

    
@register_model
def dyvit_tiny_patch16_224_teacher(pretrained=False, **kwargs):

    from models.dyvit import VisionTransformerTeacher
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = VisionTransformerTeacher(patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def dyvit_small_patch16_224(pretrained=False, **kwargs):

    from models.dyvit import DynamicVisionTransformer

    dyvit_distillation = kwargs["args"].dyvit_distill
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DynamicVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        dyvit_distillation=dyvit_distillation, distilled=deit_distillation, **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
        
    return model

   
@register_model
def dyvit_small_patch16_224_teacher(pretrained=False, **kwargs):

    from models.dyvit import VisionTransformerTeacher
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = VisionTransformerTeacher(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def dyvit_base_patch16_224(pretrained=False, **kwargs):

    from models.dyvit import DynamicVisionTransformer
    
    dyvit_distillation = kwargs["args"].dyvit_distill
        
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False
    
    model = DynamicVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        dyvit_distillation=dyvit_distillation, distilled=deit_distillation, **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    
   
@register_model
def dyvit_base_patch16_224_teacher(pretrained=False, **kwargs):

    from models.dyvit import VisionTransformerTeacher
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = VisionTransformerTeacher(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def patchmerger_tiny_patch16_224(pretrained=False, **kwargs):

    from models.patchmerger import PatchMergerVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = PatchMergerVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)


    return model
    

@register_model
def patchmerger_small_patch16_224(pretrained=False, **kwargs):

    from models.patchmerger import PatchMergerVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = PatchMergerVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

@register_model
def patchmerger_base_patch16_224(pretrained=False, **kwargs):

    from models.patchmerger import PatchMergerVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = PatchMergerVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)    

    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def sinkhorn_tiny_patch16_224(pretrained=False, **kwargs):

    from models.sinkhorn import SinkhornVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = SinkhornVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

@register_model
def sinkhorn_small_patch16_224(pretrained=False, **kwargs):

    from models.sinkhorn import SinkhornVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = SinkhornVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

@register_model
def sinkhorn_base_patch16_224(pretrained=False, **kwargs):

    from models.sinkhorn import SinkhornVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = SinkhornVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)    

    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def ats_tiny_patch16_224(pretrained=False, **kwargs):

    from models.ats import ATSVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ATSVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def ats_small_patch16_224(pretrained=False, **kwargs):

    from models.ats import ATSVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ATSVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def ats_base_patch16_224(pretrained=False, **kwargs):

    from models.ats import ATSVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ATSVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

@register_model
def heuristic_tiny_patch16_224(pretrained=False, **kwargs):

    from models.heuristic import HeuristicVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = HeuristicVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def heuristic_small_patch16_224(pretrained=False, **kwargs):

    from models.heuristic import HeuristicVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = HeuristicVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def heuristic_base_patch16_224(pretrained=False, **kwargs):

    from models.heuristic import HeuristicVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = HeuristicVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


    
@register_model
def dpcknn_tiny_patch16_224(pretrained=False, **kwargs):

    from models.dpcknn import DPCKNNVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DPCKNNVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def dpcknn_small_patch16_224(pretrained=False, **kwargs):

    from models.dpcknn import DPCKNNVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DPCKNNVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def dpcknn_base_patch16_224(pretrained=False, **kwargs):

    from models.dpcknn import DPCKNNVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DPCKNNVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def kmedoids_tiny_patch16_224(pretrained=False, **kwargs):

    from models.kmedoids import KMedoidsVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = KMedoidsVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def kmedoids_small_patch16_224(pretrained=False, **kwargs):

    from models.kmedoids import KMedoidsVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = KMedoidsVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def kmedoids_base_patch16_224(pretrained=False, **kwargs):

    from models.kmedoids import KMedoidsVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = KMedoidsVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model

    
@register_model
def topk_tiny_patch16_224(pretrained=False, **kwargs):

    from models.topk import TopKVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = TopKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def topk_small_patch16_224(pretrained=False, **kwargs):

    from models.topk import TopKVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = TopKVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def topk_base_patch16_224(pretrained=False, **kwargs):

    from models.topk import TopKVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = TopKVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

    
@register_model
def evit_tiny_patch16_224(pretrained=False, **kwargs):

    from models.evit import EfficientVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = EfficientVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def evit_small_patch16_224(pretrained=False, **kwargs):

    from models.evit import EfficientVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = EfficientVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def evit_base_patch16_224(pretrained=False, **kwargs):

    from models.evit import EfficientVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = EfficientVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def tome_tiny_patch16_224(pretrained=False, **kwargs):

    from models.tome import ToMeVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ToMeVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def tome_small_patch16_224(pretrained=False, **kwargs):

    from models.tome import ToMeVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ToMeVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def tome_base_patch16_224(pretrained=False, **kwargs):

    from models.tome import ToMeVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ToMeVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

@register_model
def sit_tiny_patch16_224(pretrained=False, **kwargs):

    from models.sit import SelfSlimmedVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = SelfSlimmedVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)


    return model
    

@register_model
def sit_small_patch16_224(pretrained=False, **kwargs):

    from models.sit import SelfSlimmedVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = SelfSlimmedVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

@register_model
def sit_base_patch16_224(pretrained=False, **kwargs):

    from models.sit import SelfSlimmedVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = SelfSlimmedVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)    

    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model