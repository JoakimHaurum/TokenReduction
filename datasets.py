import torch
from torchvision.datasets import ImageNet
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import _pil_interp
from dataloaders.nabirds import NABirds
from dataloaders.nus_wide import NUSWide

from aug_factory import CutoutPIL

try:
    from dataloaders.coco import CocoDetection
except ImportError as e:
    print("No COCO")
    print(e)

def build_dataset(root, name, split, args):
    
    if name.lower() == "imagenet":
        transform = build_imagenet_transform(split.lower() == "train", args)
        dataset = ImageNet(root=root, split=split, transform=transform)
        num_classes = 1000
    elif name.lower() == "nabirds":
        transform = build_nabirds_transform(split.lower() == "train", args)
        dataset = NABirds(root = root, train=split.lower() == "train", download=False, transform = transform)
        num_classes = len(dataset.label_map)
    elif name.lower() == "coco":
        transform = build_coco_transform(split.lower() == "train", args)
        dataset = CocoDetection(root = root, train=split.lower() == "train", transform = transform)
        num_classes = 80
    elif name.lower() == "nuswide":
        transform = build_coco_transform(split.lower() == "train", args) #Same aug approach as COCO, per ASL paper
        dataset = NUSWide(root = root, train=split.lower() == "train", transform = transform)
        num_classes = dataset.num_classes

    return dataset, num_classes

def build_coco_transform(is_train, args):

    if is_train:
        aa_transform = []
        if args.aa:
            assert isinstance(args.aa, str)
            if isinstance(args.input_size, (tuple, list)):
                img_size_min = min(args.input_size)
            else:
                img_size_min = args.input_size
            aa_params = dict(
                translate_const=int(img_size_min * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
            )
            if args.train_interpolation and args.train_interpolation != 'random':
                aa_params['interpolation'] = _pil_interp(args.train_interpolation)
            if args.aa.startswith('rand'):
                aa_transform += [rand_augment_transform(args.aa, aa_params)]
            elif args.aa.startswith('augmix'):
                aa_params['translate_pct'] = 0.3
                aa_transform += [augment_and_mix_transform(args.aa, aa_params)]
            else:
                aa_transform += [auto_augment_transform(args.aa, aa_params)]

        pre_aa = [transforms.Resize((args.input_size, args.input_size), _pil_interp(args.train_interpolation)), CutoutPIL(cutout_factor=0.5)]

        post_aa = [transforms.ToTensor(),
                   transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                        std=torch.tensor(IMAGENET_DEFAULT_STD))
                ]

        coco_transform = transforms.Compose(pre_aa + aa_transform + post_aa)

    else:
        coco_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), _pil_interp(args.train_interpolation)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD))
        ])
    
    return coco_transform

def build_nabirds_transform(is_train, args):
    
    # this should always dispatch to transforms_imagenet_train
    return create_transform(
        input_size=args.input_size,
        is_training=is_train,
        color_jitter=args.color_jitter,
        auto_augment=None,
        interpolation=args.train_interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
    )

def build_imagenet_transform(is_train, args):
    
    # this should always dispatch to transforms_imagenet_train
    return create_transform(
        input_size=args.input_size,
        is_training=is_train,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
    )

def build_imagenet_transform_deit(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=is_train,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((1.0) * args.input_size) #int((256 / 224) * args.input_size) (deit crop ratio (256 / 224), deit III crop ratio 1.0)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
