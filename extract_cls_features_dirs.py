import os
import argparse
import logging
import pandas as pd

from timm.utils import setup_default_logging
from extract_cls_features import main

_logger = logging.getLogger('validate')
    
parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data', metavar="DIR", type=str, help='dataset path')
parser.add_argument('--dataset', '-d', metavar='NAME', default='imagenet', choices=['imagenet', 'nabirds', "coco", "nuswide"], type=str, help='Dataset to evlauate on')

parser.add_argument("--dataset_csv", default="", type=str)
parser.add_argument("--parent_dir", default=".", type=str)
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use_amp', action='store_true', help="")
parser.add_argument('--device', default='cuda', help='device to use for training / testing')

parser.add_argument('--viz_mode', action='store_true', help="")
parser.add_argument('--overwrite_existing', action='store_true', help="")


def dir_main():
    setup_default_logging()
    args = parser.parse_args()

    dataset_csv = args.dataset_csv
    output_dir = args.output_dir
    parent_dir = args.parent_dir

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_csv, sep=",")

    df["Full Path"] = df.apply(lambda r: os.path.join(r["output_dir"], r["Name"]), axis=1) 

    for index, row in df.iterrows():

        full_dir_path = parent_dir + row['Full Path'][1:]
        print(full_dir_path, row["model"])

        if not os.path.isdir(full_dir_path):
            continue

        full_path = os.path.join(full_dir_path, "best_checkpoint.pth")
        if not os.path.isfile(full_path):
            continue
        
        args.viz_output_name = row["Name"]+"_cls_features"
        
        if os.path.isfile(os.path.join(output_dir, args.viz_output_name)) and not args.overwrite_existing:
            continue

        args.checkpoint = full_dir_path
        args.output_dir = output_dir

        try:
            main(args, _logger)
        except Exception as e:
            _logger.error(e)
               

if __name__ == '__main__':
    dir_main()
