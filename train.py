# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path
from contextlib import suppress

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, AsymmetricLossMultiLabel
from scheduler_factory import create_scheduler
from optim import create_optimizer
from timm.utils import get_state_dict, ModelEmaV2
from mp_scaler import NativeScalerGradAcum

from datasets import build_dataset
from engine import train_one_epoch, evaluate_multiclass, evaluate_multilabel
from losses import DistillationLoss, DynamicViTDistillationLoss
from samplers import RASampler

import models_act
import utils
import wandb
wandb.login()


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'nabirds', "coco", "nuswide"],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    parser.add_argument("--wandb_project", default="Token Reduction Training", type=str)
    parser.add_argument("--wandb_group", default="MISC", type=str)

    parser.add_argument('--backbone_lr_scale', default=1.0, type=float, help="")
    parser.add_argument('--backbone_freeze_steps', default=0, type=int, help="")
    parser.add_argument('--constant_cls', action='store_true', help="")
    parser.add_argument('--constant_pos', action='store_true', help="")


    parser.add_argument('--use_amp', action='store_true', help="")
    parser.add_argument('--sched_in_steps', action='store_true', help="")
    parser.add_argument('--grad_accum_steps', default = 1, type=int, help="")
    parser.add_argument('--lr_batch_normalizer', default = 512, type=float, help="")
    
    parser.add_argument('--save_more_than_best', action='store_true', help="")

    temp_args, _ = parser.parse_known_args()
    
    parser.add_argument('--reduction_loc', type=int, nargs='+', default=[]) 
    parser.add_argument('--keep_rate', type=float, nargs='+', default=[])
    
    if "dyvit" in temp_args.model.lower():
        parser.add_argument('--token_distill_weight', default=0.5, type=float)
        parser.add_argument('--cls_distill_weight', default=0.5, type=float)
        parser.add_argument('--ratio_weight', default=2.0, type=float)
        parser.add_argument('--cls_weight', default=1.0, type=float)   
        parser.add_argument('--mse_token', action='store_true') 
        parser.add_argument('--dyvit_distill', action='store_true') 
        parser.add_argument('--no_dyvit_teacher', action='store_true') 
        parser.add_argument('--dyvit_teacher_weights', default="", type=str)
        parser.set_defaults(dyvit_distill=True)
        parser.set_defaults(mse_token=True)

    if "dpcknn" in temp_args.model.lower():
        parser.add_argument('--k_neighbors', default=5, type=int)
        
    if "heuristic" in temp_args.model.lower():
        parser.add_argument('--heuristic_pattern', type=str, default="l1", choices={"l1", "l2", "linf"})                        
        parser.add_argument('--min_radius', type=float, default=1.0) 
        parser.add_argument('--not_contiguous', action='store_true') 
    
    if "sinkhorn" in temp_args.model.lower():
        parser.add_argument('--sinkhorn_eps', type=float, default=1.0)   
        
    if "kmedoids" in temp_args.model.lower() or "sinkhorn" in temp_args.model.lower():
        parser.add_argument('--cluster_iters', type=int, default=3)   
        
    if "kmedoids" in temp_args.model.lower() or "dpcknn" in temp_args.model.lower():
        parser.add_argument('--equal_weight', action='store_true') 

    return parser


def main(args):

    utils.init_distributed_mode(args)

    print(utils.is_main_process(), utils.get_rank())
 
    args.total_batch_size = args.batch_size * args.grad_accum_steps * utils.get_world_size()

    if utils.is_main_process() and not args.eval:
        wandb.init(project=args.wandb_project, group = args.wandb_group, config=args)

    if args.output_dir and utils.is_main_process():
        args.output_dir = os.path.join(args.output_dir, wandb.run.name)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.num_classes = build_dataset(args.data, args.dataset, "train", args=args)
    dataset_val, _ = build_dataset(args.data, args.dataset, "val", args=args)

    if utils.is_main_process():
        print(args.dataset, args.num_classes, len(dataset_train), len(dataset_val))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        args = args
    )

    if args.dataset.lower() != "imagenet":
        model.reset_classifier(args.num_classes)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            pass
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            pass
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if utils.is_main_process():
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

    # Scale LR and create Optimizer
    args.num_steps_epoch = len(dataset_train) // args.total_batch_size
    
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.total_batch_size / args.lr_batch_normalizer
        args.input_lr = args.lr
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    # Setup 16-bit training if chosen
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.use_amp:
        loss_scaler = NativeScalerGradAcum()
        amp_autocast = torch.cuda.amp.autocast

    # Create LR Scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Create Loss function
    if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
        criterion = LabelSmoothingCrossEntropy()
        if mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        print("Using ASL Loss")
        criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=False)
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # Setup teacher_model for Deit Distillation
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=1000,
            global_pool='avg',
            args = args
        )
        
        if teacher_model is not None and args.dataset.lower() != "imagenet":
            teacher_model.reset_classifier(args.num_classes)

        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # Setup teacher model for DynamicViT distillation
    if "dyvit" in args.model:
        if args.no_dyvit_teacher:
            teacher_model = None
        else:          
            teacher_model = create_model(args.model+"_teacher",
                            pretrained = True,
                            num_classes=1000,
                            drop_rate=args.drop,
                            drop_path_rate=args.drop_path,
                            drop_block_rate=None,
                            img_size=args.input_size,
                            args = args
            )
            
            if args.dataset.lower() != "imagenet":
                assert(args.dyvit_teacher_weights != ""), "Empty DyViT Teacher Weight path"
                assert(os.path.isfile(args.dyvit_teacher_weights)), "Invalid DyViT Teacher Weight path: {}".format(args.dyvit_teacher_weights)

                teacher_model.reset_classifier(args.num_classes)
                
                checkpoint = torch.load(args.dyvit_teacher_weights, map_location='cpu')

                if checkpoint["ema_best"]:
                    teacher_model.load_state_dict(checkpoint['model_ema'])
                else:
                    teacher_model.load_state_dict(checkpoint['model'])

            teacher_model.to(device)
            teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    if "dyvit" in args.model:
        criterion = DynamicViTDistillationLoss(
            criterion, teacher_model, args.ratio_weight, args.cls_distill_weight, args.token_distill_weight, args.cls_weight, args.mse_token)
    else:
        criterion = DistillationLoss(
            criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
        )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        if utils.is_main_process():
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    if args.dataset.lower() == "coco" or args.dataset.lower() == "nuswide":
        evaluate = evaluate_multilabel
    else:
        evaluate = evaluate_multiclass


    test_stats = evaluate(data_loader_val, model, device)
    
    if utils.is_main_process():
        max_accuracy = test_stats["acc1"]
        log_stats = {**{f'val_{k}': v for k, v in test_stats.items()},
                        "max_accuracy": max_accuracy}
        
        if model_ema is not None and not args.model_ema_force_cpu:
            log_stats = {**log_stats,
                         **{f'ema_val_{k}': v for k, v in test_stats.items()}}
        status_print = f"Epoch: 0\tDataset: {len(dataset_val)}\tAcc@1: {test_stats['acc1']:.1f}%"
        print(status_print)       

        wandb.log(log_stats, step=0)      

    if utils.is_main_process():
        print(f"Start training for {args.epochs} epochs")
        wandb.watch(model)
        
    start_time = time.time()
    max_accuracy = 0.0

    if args.epochs+args.cooldown_epochs == 0:
        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / 'best_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': 0,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                    'args': args,
                    'ema_best': False,
                }, checkpoint_path)
    else:                
        for epoch in range(args.start_epoch, args.epochs+args.cooldown_epochs):
            
            for param_group in optimizer.param_groups:
                if epoch == 0:
                    for p in param_group["params"]:
                        p.grad =  None
                if epoch == param_group['fix_step']:
                    for p in param_group["params"]:
                        p.grad = torch.zeros_like(p)
                    
                        
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats, total_step = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler, lr_scheduler,
                args.clip_grad, model_ema, mixup_fn, amp_autocast,
                set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                grad_accum_steps = args.grad_accum_steps,
                num_steps_epoch = args.num_steps_epoch,
                print_freq = args.num_steps_epoch // 5 if args.num_steps_epoch // 5 > 0 else 1,
                multi_label = args.dataset.lower() == "coco" or args.dataset.lower() == "nuswide"
            )

            lr_scheduler.step(epoch+1)
            if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, checkpoint_path)
                

            test_stats = evaluate(data_loader_val, model, device, amp_autocast)
            
            max_accuracy_flag = False

            if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, checkpoint_path)

            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                max_accuracy_flag = True
                if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                    checkpoint_paths = [output_dir / 'best_standard_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                            'args': args,
                        }, checkpoint_path)


            ema_test_states = None
            if model_ema is not None and not args.model_ema_force_cpu:
                ema_test_states = evaluate(data_loader_val, model_ema.module, device, amp_autocast)
            
                if max_accuracy < ema_test_states["acc1"]:
                    max_accuracy = ema_test_states["acc1"]
                    max_accuracy_flag = True
                    if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                        checkpoint_paths = [output_dir / 'best_ema_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'model_ema': get_state_dict(model_ema),
                                'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                                'args': args,
                            }, checkpoint_path)


            if max_accuracy_flag:
                ema_best = False
                if model_ema is not None:
                    ema_best = ema_test_states["acc1"] > test_stats["acc1"]

                if args.output_dir and utils.is_main_process():
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                            'args': args,
                            'ema_best': ema_best,
                        }, checkpoint_path)
                 
            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in test_stats.items()},
                            "max_accuracy": max_accuracy}
                status_print = f"Epoch: {epoch}\tDataset: {len(dataset_val)}\tAcc@1: {test_stats['acc1']:.1f}%"

                if ema_test_states is not None:
                    status_print += f"\tEMA-Acc@1: {ema_test_states['acc1']:.1f}%"
                    log_stats = {**log_stats,
                                **{f'ema_val_{k}': v for k, v in ema_test_states.items()}}
                                
                status_print += f"\tMax Acc@1: {max_accuracy:.2f}%"
                print(status_print)       

                wandb.log(log_stats, step=total_step)
      
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        print('Training time {}'.format(total_time_str))
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
