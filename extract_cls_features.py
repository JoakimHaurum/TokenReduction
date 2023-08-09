
import argparse
import os
import time
import torch

import numpy as np

from contextlib import suppress
from timm.models import create_model

from datasets import build_dataset
import models_act

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data', metavar="DIR", type=str, help='dataset path')
parser.add_argument('--dataset', '-d', metavar='NAME', default='imagenet', choices=['imagenet', 'nabirds', "coco", "nuswide"], type=str, help='Dataset to evlauate on')
parser.add_argument('--split', metavar='NAME', default='validation', help='dataset split (default: validation)')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use_amp', action='store_true', help="")
parser.add_argument('--device', default='cuda', help='device to use for training / testing')

parser.add_argument('--viz_mode', action='store_true', help="")

def validate(args, _logger):
    amp_autocast = suppress  # do nothing
    if args.use_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')

    assert args.checkpoint != "", "Empty checkpoint path, not usable"
    assert os.path.isdir(args.checkpoint), "Checkpoint path is not dir, not usable: {}".format(args.checkpoint)
    assert os.path.isfile(os.path.join(args.checkpoint, "best_checkpoint.pth")), "Checkpoint path does not have a 'best_checkpoint.pth' file"
           

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Setting for posterity
    args.color_jitter = 0
    args.aa = ""
    args.train_interpolation = "bicubic"
    args.reprob = 0
    args.remode = ""
    args.recount = 0

    dataset_val, args.num_classes = build_dataset(args.data, args.dataset, "val", args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    
    checkpoint = torch.load(os.path.join(args.checkpoint, "best_checkpoint.pth"), map_location='cpu')
    model_args = checkpoint["args"]

    model_name = model_args.model

    if "deit" in model_name:
        model_name += "_viz"

    _logger.info(f"Creating model: {model_name}")
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=args.num_classes,
        img_size=model_args.input_size,
        args = model_args
    )
    model.viz_mode = args.viz_mode

    if checkpoint["ema_best"]:
        model.load_state_dict(checkpoint['model_ema'])
    else:
        model.load_state_dict(checkpoint['model'])

    
    _logger.info("counting parameters")

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("logging")
    _logger.info('Model %s created, param count: %d' % (model_args.model, param_count))

    _logger.info("moving to device")
    model.to(device)
    model.eval()

    _logger.info("Setting up Loss")
    
    model.eval()

    _logger.info("Ready for Inference")

    start = time.time()
    with torch.no_grad():
        end = time.time()

        feature_matrices = {3: None, 6: None, 9: None, 11: None}

        for batch_idx, (input, target) in enumerate(data_loader_val):
            target = target.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)

            # compute output
            with amp_autocast():
                output = model(input)

            if args.viz_mode:
                output, viz_data = output 


            for key in list(feature_matrices.keys()):

                if feature_matrices[key] is None:
                    feature_matrices[key] = viz_data["Features"][key][:,0]     
                else:
                    feature_matrices[key] = np.vstack([feature_matrices[key], viz_data["Features"][key][:,0]])


            # measure elapsed time
            elapsed_time = time.time() - end
            end = time.time()
            
            if batch_idx % 50 == 0:
                print("Batch time: {}\t Total time: {}".format(elapsed_time, end-start))
    return feature_matrices

def main(args, _logger):  
    
    viz_data = validate(args, _logger)
    viz_data_file = os.path.join(args.output_dir, args.viz_output_name)
    write_viz(viz_data_file, viz_data)

def write_viz(viz_file, viz_data):

    for key in list(viz_data.keys()):
        print(viz_file+"_"+str(key)+".npy", viz_data[key].T.shape)
        np.save(viz_file+"_"+str(key)+".npy", viz_data[key].T)

if __name__ == '__main__':

    from timm.utils import setup_default_logging
    import logging

    _logger = logging.getLogger('validate')
    setup_default_logging()
    args = parser.parse_args()
    main(args, _logger)
