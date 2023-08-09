# Which Tokens to Use? Investigating Token Reduction in Vision Transformers

This repository is the official implementation of [Which Tokens to Use? Investigating Token Reduction in Vision Transformers](). 

We conduct the first systematic comparison and analysis of 10 state-of-the-art token reduction methods across four image classification datasets, trained using a single codebase and consistent training protocol. Through extensive experiments providing deeper insights into the core mechanisms of the token reduction methods. We make all of the training and analysis code used for the project available.

The project page can be found [here](http://vap.aau.dk/tokens).

## Requirements

Requirements can be found in the requirements.txt file. 

## Datasets

We test on the ImageNet-1K, NABirds, COCO 2014, and NUS-Wide datasets. They are available through the following links:

- Imagenet is available through Kaggle: [https://www.kaggle.com/c/imagenet-object-localization-challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge)
- NABirds is available through the official website: [https://dl.allaboutbirds.org/nabirds](https://dl.allaboutbirds.org/nabirds)
- COCO is available through the official website: [https://cocodataset.org/](https://cocodataset.org/)
- For NUS-Wide, we use the variation made available by Alibaba-MIIL: [https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md](https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md)


## Training

The models are trained by finetuning a pretrained DeiT model. Reduction blocks are inserted using the `--reduction_loc` argument. In our paper we focus on reducing at the 4th, 7th, and 10th block (note that the argument is 0 indexed), as commonly done in the litterature. In order to work with a larger effective batch size than what can be fitted on the GPU VRAM, we use the `--grad_accum_steps` argument to aggregate over batches.

An example training command is shown below, training on ImageNet with the Top-K reduction method and a keep rate of 0.9. Note that for merging based methods the keep-rate argument denotes the number of clusters, while for the ATS method the keep rate is the maximum number of tokens kept.

```
train.py --dataset imagenet --data <path_to_data> --batch-size 256 --lr 0.001 --epochs 30 --warmup-epochs 20 --lr_batch_normalizer 1024 --sched_in_steps --use_amp --grad_accum_steps 2 --wandb_project <wandb_project> --wandb_group <wandb_group> --output_dir <path_to_model_output> --model topk_small_patch16_224 --reduction_loc 3 6 9 --keep_rate 0.9
```

## Evaluation

In order to evaluate the trained models, we first extract an overview of the Weights and Biases (W&B) project where we have logged the training runs.

```
get_wandb_tables.py --entity <wandb_entity> --project <wandb_project> --output_path <path_to_wandb_overviews>
```

Using the W&B overview, we extract the metric performance as well as reduction patterns of all models in the W&B project if the viz_mode argument is provided.

```
validate_dirs.py --dataset imagenet --data <path_to_data> --output_dir <path_to_eval_output> --dataset_csv <path_to_wandb_overviews> --viz_mode --use_amp
```

## Analysis

In our analysis we investigate how reduction patterns and CLS Features compares across trained models.

### Comparing CLS Features and per-image Reduction Patterns 

The reduction pattern analysis is conducted using the scripts in the reduction_methods_analysis directory.
CLS features and reduction patterns are compare within models across keep rates and model capacities using the `compare_{cls_features, merging, pruning}_rates.py` and `compare_{cls_features, merging, pruning}_capacity.py` scripts, respectively. In order to compare across reduction methods we use the `compare_{cls_features, merging, pruning}_models.py` scripts.

The scripts take 4 common arguments:
- `parrent_dir`: The path to where the extracted reduction patterns (using validate.py) are stored
- `dataset_csv`: The path to the extracted W&B overview csv file
- `output_file`: Name of the output file
- `output_path`: Path to the where the output_file will be saved

The `*_models.py` scripts also take a `capacity` argument to indicate which model capacities to compare.

In order to compare CLS features, we need to extract them using the `extract_cls_features_dirs.py` script, which shares the same arguments as the `validate_dirs.py` script.

### Extracting and Comparing Global Pruning-based Reduction Patterns

In order to compare reduction patterns across datasets we first determine the average depth of each token. This is extracted using the `compute_token_statistics.py` script, which shares arguments with the previous compare scripts.

Using the extracted token statistics the global pruning-based reduction patterns can be compared using the `compare_heatmaps.py` script.
Per default the script simply compares the same reduction method at the same capacity and keep rate across datasets. This cna be modified using hte following arguments:
- `compare_within_dataset`
- `compare_across_rates`
- `compare_across_capacities`
- `compare_across_models`

### Collating data and calculating correlations

The output data of the compare scripts are collated using the corresponding `collate_{capacity, models, rates}_data.py` scripts. The scripts assumes the cls_features, merging, and pruning output files from the compare scripts follow a specific naming scheme. See the scripts for details.

The collated files are then used to compute the Spearman's and Kendall's correlations coefficients between the difference in metric performance and the different reduction pattern and CLS feature similaritiy metrics. This is done using the `calculate_correlation_{capacity, models, rates}.py` scripts.



## Code Credits

The token reduction method code is based and inspired by:
- The DPC-KNN method is based on the official code repository: [https://github.com/zengwang430521/TCFormer](https://github.com/zengwang430521/TCFormer)
- The DynamicViT method is based on the official code repository: [https://github.com/raoyongming/DynamicViT](https://github.com/raoyongming/DynamicViT)
- The EViT and Top-K methods are based on the official EViT code repository: [https://github.com/youweiliang/evit](https://github.com/youweiliang/evit)
- The Sinkhorn method is based on the official code repository: [https://github.com/JoakimHaurum/MSHViT](https://github.com/JoakimHaurum/MSHViT)
- The SiT method is based on the official code repository: [https://github.com/Sense-X/SiT](https://github.com/Sense-X/SiT)
- The ToMe method is based on the official code repository: [https://github.com/facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)
- The ATS and PatchMerger methods are based on the implementation from Phil Wang (Lucidrains): [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)

Parts of the training code and large part of the ViT implementation is based and inspired by:
- The timm framework by Ross Wrightman / HuggingFace: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- The DeiT training and evaluation code: [https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)
- The EViT training and evaluation code: [https://github.com/youweiliang/evit](https://github.com/youweiliang/evit)
- The multi-label training and evaluation code from the Assymetric Loss paper: [https://github.com/Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL)

Parts of the analysis code is based and inspiredby:
- The model feature similarity metrics are based on the official code repository of the "Grounding Representation Similarity with Statistical Testing" paper: [https://github.com/js-d/sim_metric](https://github.com/js-d/sim_metric)
- The metrics used for comparing global pruning-based reduction pattern are ported from the official MATLAB code repository of the "What do different evaluation metrics tell us about saliency models?" paper: [https://github.com/cvzoya/saliency/tree/master/code_forMetrics](https://github.com/cvzoya/saliency/tree/master/code_forMetrics)

## License

The Code is licensed under an MIT License, with exceptions of the afforementioned code credits which follows the license of the original authors.

## Bibtex
```bibtex
@article{Haurum_2023_ICCVW,
author = {Joakim Bruslund Haurum and Sergio Escalera and Graham W. Taylor and Thomas B. Moeslund},
title = {Which Tokens to Use? Investigating Token Reduction in Vision Transformers},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
month = {October},
year = {2023},
}
```