import argparse
import os
import numpy as np
import pandas as pd

import reduction_methods_analysis.feature_sim_metrics as feature_metrics
from reduction_methods_analysis.analysis_utils import get_model_pair

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--parent_dir', default="", type=str)
parser.add_argument("--dataset_csv", default="", type=str)
parser.add_argument("--output_file", default="", type=str)
parser.add_argument("--output_dir", default="", type=str)

def compare_main():
    args = parser.parse_args()
    model_parent_dir = args.parent_dir
    overview_path = args.dataset_csv
    output_file = args.output_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    overview_df = pd.read_csv(overview_path, sep=",")    
    overview_df["Model Capacity"] = overview_df.apply(lambda r: (r["model"].split("_")[1]), axis=1)
    overview_df["Model Reduced"] = overview_df.apply(lambda r: ("linf" + "-" + r["model"] ) if "linf" in r["model"] else r["model"], axis=1)
    overview_df["Model Reduced"] = overview_df.apply(lambda r: r["Model Reduced"].split("_")[0], axis=1)
    overview_df["model"] = overview_df.apply(lambda r: (r["model"] + "-" + r["heuristic_pattern"]) if "heuristic" in r["model"] else r["model"], axis=1)
    overview_df = overview_df.sort_values(by=["Model Reduced","Model Capacity"], ascending=[True, False])
    print(overview_df)

    comp_locs = [3,6,9,11]

    output = pd.DataFrame()
    model_pairs = []
    file_pairs = []
    for index1, row1 in overview_df.iterrows():
        name1 = row1["Name"]
        model1 = row1["model"]
        
        model1_name_reduced = row1["Model Reduced"]
        model1_capacity = row1["Model Capacity"]

        if model1_capacity == "tiny":
            continue

        ratio1 = row1["keep_rate"].replace("[","").replace("]","")
        if ratio1 != "":
            ratio1 = float(ratio1)
        loc1 = row1["reduction_loc"].replace("[","").replace("]","")
        if loc1 != "":
            loc1 = [int(idx) for idx in loc1.split(",")]

        file1 = name1 + "_cls_features"

        model1_name = get_model_pair(model1, loc1, ratio1)
        print(model1_name)
        
        for index2, row2 in overview_df.iterrows():          
            name2 = row2["Name"]
            model2 = row2["model"]
            
            model2_name_reduced = row2["Model Reduced"]
            model2_capacity = row2["Model Capacity"]

            if model1_name_reduced != model2_name_reduced:
                continue
            
            if model1_capacity == "base":
                if model2_capacity == "base":
                    continue
            elif model1_capacity == "small":
                if model2_capacity == "base" or model2_capacity == "small":
                    continue

            ratio2 = row2["keep_rate"].replace("[","").replace("]","")
            if ratio2 != "":
                ratio2 = float(ratio2)
            loc2 = row2["reduction_loc"].replace("[","").replace("]","")
            if loc2 != "":
                loc2 = [int(idx) for idx in loc2.split(",")]


            if ratio1 != ratio2:
                continue

            if loc1 != loc2:
                continue

            file12 = name1+"+"+name2
            file21 = name2+"+"+name1

            if file12 in file_pairs or file21 in file_pairs:
                continue

            file2 = name2 + "_cls_features"

            if file1 == file2:
                continue
                    
            model2_name = get_model_pair(model2, loc2, ratio2)

            comb_model_name12 = model1_name+"+"+model2_name
            comb_model_name21 = model2_name+"+"+model1_name

            if comb_model_name12 in model_pairs or comb_model_name21 in model_pairs:
                continue
            else:
                model_pairs.append(comb_model_name12)
                model_pairs.append(comb_model_name21)    
                file_pairs.append(file12)
                file_pairs.append(file21)  

            print("\t{}".format(model2_name))

            res_dict = {"Model A": model1_name,
                        "Model B": model2_name,
                        "Ratio": ratio1,
                        "Loc": loc1}

            for comp_loc in comp_locs:
                path1 = os.path.join(model_parent_dir, file1+"_{}.npy".format(comp_loc))
                path2 = os.path.join(model_parent_dir, file2+"_{}.npy".format(comp_loc))

                data1 = np.load(path1)
                data1 = data1 - data1.mean(axis=1, keepdims=True)
                data1 = data1 / np.linalg.norm(data1)

                data2 = np.load(path2)
                data2 = data2 - data2.mean(axis=1, keepdims=True)
                data2 = data2 / np.linalg.norm(data2)

                metric_dict = {}
                            
                _, cca_rho, _, transformed_rep1, _ = feature_metrics.cca_decomp(data1, data2)
                metric_dict["{}-PWCCA".format(comp_loc)] = feature_metrics.pwcca_dist(data1, cca_rho, transformed_rep1)
                metric_dict["{}-mean_sq_cca_corr".format(comp_loc)] = feature_metrics.mean_sq_cca_corr(cca_rho)
                metric_dict["{}-mean_cca_corr".format(comp_loc)] = feature_metrics.mean_cca_corr(cca_rho)

                metric_dict["{}-CKA".format(comp_loc)] = feature_metrics.lin_cka_dist(data1, data2)
                metric_dict["{}-Procrustes".format(comp_loc)] = feature_metrics.procrustes(data1, data2)

                res_dict = {**res_dict,
                            **metric_dict}

            output = output.append(res_dict, ignore_index=True)
        output.to_csv(os.path.join(output_dir, output_file), sep=";", index=False)



if __name__ == '__main__':
    compare_main()
