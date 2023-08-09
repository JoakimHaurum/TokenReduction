
import argparse
import os
import time
import numpy as np
import pandas as pd

import heatmap_metrics

try:
    import ujson as json
    print("Using ujson")
except ImportError:
    try:
        import simplejson as json
        print("Using simplejson")
    except ImportError:
        import json
        print("Using json")

def check_dataset(path):
    if "IM" in path:
        return "IM"
    if "NAB" in path:
        return "NAB"
    if "COCO" in path:
        return "COCO"
    if "NUS" in path:
        return "NUS"

parser = argparse.ArgumentParser('Comapre pruning-based reduction patterns', add_help=False)
parser.add_argument('--dataset_IM', default="", type=str)
parser.add_argument('--dataset_NAB', default="", type=str)
parser.add_argument('--dataset_COCO', default="", type=str)
parser.add_argument('--dataset_NUS', default="", type=str)
parser.add_argument('--input_path', default="", type=str)
parser.add_argument('--output_path', default=".", type=str)
parser.add_argument('--compare_within_dataset', action="store_true")
parser.add_argument('--compare_across_rates', action="store_true")
parser.add_argument('--compare_across_capacities', action="store_true")
parser.add_argument('--compare_across_models', action="store_true")
args = parser.parse_args()

datasets = [args.dataset_IM, args.dataset_NAB, args.dataset_COCO, args.dataset_NUS]
checked_dataset = []

emd_dist = heatmap_metrics.create_emd_dist(14, 14)

output = pd.DataFrame()
for dataset1 in datasets:
    path1 = os.path.join(args.input_path, dataset1)
    with open(path1) as json_file:
        data1 = json.load(json_file)

    dataset1_tag = check_dataset(path1)
    for dataset2 in datasets:
        start = time.time()

        if dataset1+dataset2 in checked_dataset or dataset2+dataset1 in checked_dataset:
            continue
        else:
            checked_dataset.append(dataset1+dataset2)
            checked_dataset.append(dataset2+dataset1)

        if not args.compare_within_dataset and dataset1 == dataset2:
            continue

        if dataset1 == dataset2:
            data2 = data1
            dataset2_tag = dataset1_tag
        else:
            path2 = os.path.join(args.input_path, dataset2)
            dataset2_tag = check_dataset(path2)
            with open(path2) as json_file:
                data2 = json.load(json_file)

        models = set(list(data1.keys())).intersection(list(data2.keys()))
        models = [x for x in models if not any(y in x for y in ["sinkhorn", "patchmerger", "heuristic", "sit", "tome", "deit"])]
        checked_pair = []

        for model1 in models:
            
            pattern1 = np.reshape(data1[model1]["Mean-Token-Depth-Spatial"], (14,14))
            model1_split = model1.split("_")


            for model2 in models:
                model2_split = model2.split("_")

                if model1+model2 in checked_pair or model2+model1 in checked_pair:
                    continue
                if not args.compare_across_models and model1_split[0] != model2_split[0]:
                    continue
                if not args.compare_across_capacities and model1_split[1] != model2_split[1]:
                    continue
                if not args.compare_across_rates and data1[model1]["Ratio"] != data2[model2]["Ratio"]:
                    continue
                
                pattern2 = np.reshape(data2[model2]["Mean-Token-Depth-Spatial"], (14,14))
                df_dict = {}
                df_dict["Dataset1"] = dataset1_tag
                df_dict["Dataset2"] = dataset2_tag
                df_dict["Model1"] = model1
                df_dict["Model2"] = model2
                df_dict["KLD12"] = heatmap_metrics.KL(pattern1, pattern2)
                df_dict["KLD21"] = heatmap_metrics.KL(pattern2, pattern1)
                df_dict["JSD"] = heatmap_metrics.JS(pattern1, pattern2)
                df_dict["PCC"] = heatmap_metrics.PCC(pattern1, pattern2)
                df_dict["SCC"] = heatmap_metrics.SCC(pattern1, pattern2)
                df_dict["EMD"] = heatmap_metrics.EMD(pattern1, pattern2, emd_dist)
                df_dict["SIM"] = heatmap_metrics.SIM(pattern1, pattern2)
                
                output = output.append(df_dict, ignore_index=True)

                checked_pair.append(model1+model2)
                checked_pair.append(model2+model1)
    print(checked_dataset)
output = output.sort_values(by=["Dataset1", "Dataset2", 'Model1', "Model2"])
output.to_csv(os.path.join(args.output_path, "heatmap_comparison.csv"), sep=";", index=False)
