import argparse
import os
import numpy as np
import pandas as pd

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

parser = argparse.ArgumentParser(description='Collate data computed across capacities for the same reduction method(s)')
parser.add_argument('--parent_dir', default="", type=str)
parser.add_argument("--datasets", nargs='+', type=str, default = ["IM","NAB","COCO","NUS"])
parser.add_argument("--output_dir", default="", type=str)

def collate_capacity_data():
    args = parser.parse_args()
    model_parent_dir = args.parent_dir
    datasets = args.datasets
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    stages = [3, 6, 9, 11]
    base_cols = ["Model A", "Model B", "Ratio"]

    corr_col_tags_base = ["CKA", "PWCCA", "Procrustes", "mean_cca_corr", "mean_sq_cca_corr"]
    cluster_col_tags_base = ["Completeness", "Homogeneity", "NMI"]
    pruning_col_tags_base = ["IoA", "IoU"]


    corr_col_tags = ["{}-{}".format(str(stage), tag) for stage in stages for tag in corr_col_tags_base]
    cluster_col_tags = ["{}-Stage-{}-Mean".format(tag, str(stage)) for stage in stages[:3] for tag in cluster_col_tags_base]
    pruning_col_tags = ["Stage-{}-Mean-{}".format(str(stage), tag) for stage in stages[:3] for tag in pruning_col_tags_base]

    print(corr_col_tags)
    print(cluster_col_tags)
    print(pruning_col_tags)

    for dataset in datasets:
        token_stats_path = os.path.join(model_parent_dir, f"token_stats_{dataset}.json")
        pruning_path = os.path.join(model_parent_dir, f"pruning_comparison_capacity_{dataset}.csv")
        cluster_path = os.path.join(model_parent_dir, f"cluster_comparison_capacity_{dataset}.csv")
        corr_path = os.path.join(model_parent_dir, f"cls_features_comparison_capacity_{dataset}.csv")
        
        with open(token_stats_path) as json_file:
            token_stats_data = json.load(json_file)

        pruning_data = pd.read_csv(pruning_path, sep=";")
        cluster_data = pd.read_csv(cluster_path, sep=";")
        corr_data = pd.read_csv(corr_path, sep=";")
        
        pruning_data_subset = pruning_data[base_cols + pruning_col_tags]
        cluster_data_subset = cluster_data[base_cols + cluster_col_tags]
        corr_data_subset = corr_data[base_cols + corr_col_tags]
        
        tags_dict = {key:[] for key in corr_col_tags+cluster_col_tags+pruning_col_tags}
        total_data_dict = {"Model A": [], "Model B": [], "Ratio": [], "Acc A": [], "Acc B": [], "Acc Diff": [], **tags_dict}
        
        for idx, row in corr_data.iterrows():
            model_a = row["Model A"]
            model_b = row["Model B"]
            total_data_dict["Model A"].append(row["Model A"])
            total_data_dict["Model B"].append(row["Model B"])
            total_data_dict["Ratio"].append(row["Ratio"])
            total_data_dict["Acc A"].append(token_stats_data[model_a]["Acc-Top1"])
            total_data_dict["Acc B"].append(token_stats_data[model_b]["Acc-Top1"])
            total_data_dict["Acc Diff"].append(token_stats_data[model_a]["Acc-Top1"] - token_stats_data[model_b]["Acc-Top1"])

            pruning_data_row = pruning_data_subset[(pruning_data_subset["Model A"] == model_a) & (pruning_data_subset["Model B"] == model_b)]
            cluster_data_row = cluster_data_subset[(cluster_data_subset["Model A"] == model_a) & (cluster_data_subset["Model B"] == model_b)]
            corr_data_row = corr_data_subset[(corr_data_subset["Model A"] == model_a) & (corr_data_subset["Model B"] == model_b)]

            for tag in pruning_col_tags:
                if len(pruning_data_row[tag]) > 0:
                    total_data_dict[tag].append(pruning_data_row[tag].values[0])
                else:
                    total_data_dict[tag].append(np.nan)

            for tag in cluster_col_tags:
                if len(cluster_data_row[tag]) > 0:
                    total_data_dict[tag].append(cluster_data_row[tag].values[0])
                else:
                    total_data_dict[tag].append(np.nan)

            for tag in corr_col_tags:
                if len(corr_data_row[tag]) > 0:
                    total_data_dict[tag].append(corr_data_row[tag].values[0])
                else:
                    total_data_dict[tag].append(np.nan)

        result_df = pd.DataFrame(data=total_data_dict)
        result_df = result_df.sort_values(by=['Model A', 'Model B'])
        
        result_df.to_csv(os.path.join(output_dir, f"collated_comparison_capacity_{dataset}.csv"), sep=";", index=False)
        print(result_df)

if __name__ == '__main__':
    collate_capacity_data()