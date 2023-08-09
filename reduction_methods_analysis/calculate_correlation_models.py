import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser(description='Correlated data computed across reduction methods with difference in metric performance')
parser.add_argument('--parent_dir', default="", type=str)
parser.add_argument("--datasets", nargs='+', type=str, default = ["IM","NAB","COCO","NUS"])
parser.add_argument("--capacities", nargs='+', type=str, default = ["base", "small", "tiny"])
parser.add_argument("--output_dir", default="", type=str)

def correlate_capacity_data():
    args = parser.parse_args()
    base_path = args.parent_dir
    datasets = args.datasets
    capacities = args.capacities
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        for capacity in capacities:
            path = f"collated_comparison_models_{capacity}_{dataset}.csv"

            df = pd.read_csv(os.path.join(base_path, path), sep=";")
            df["Model Reduced A"] = df.apply(lambda r: (r["Model A"].split("_")[0]), axis=1)
            df["Model Reduced B"] = df.apply(lambda r: (r["Model B"].split("_")[0]), axis=1)
            
            df["Ratio B"] = df["Ratio B"].replace(np.nan, '1.0')
            df["Ratio B"] = df["Ratio B"].astype(str)

            columns = list(df.columns)
            columns.remove("Model A")
            columns.remove("Model B")
            columns.remove("Ratio A")
            columns.remove("Ratio B")
            columns.remove("Acc A")
            columns.remove("Acc B")
            columns.remove("Acc Diff")
            columns.remove("Model Reduced A")
            columns.remove("Model Reduced B")

            unique_models = df["Model A"].unique()
            output = pd.DataFrame()
            
            for model in unique_models:
                if "deit" in model.lower():
                    for ratio in ["0.25", "0.5", "0.7", "0.9"]:
                        model_df = df[df["Model A"] == model]

                        ratios = [ratio]
                        if ratio == "0.9":
                            ratios.append("1.0")

                        model_df = model_df[model_df["Ratio B"].isin(ratios)]
                        acc_diff = np.asarray(model_df["Acc Diff"].values)
                        results = {"Model": model+ratio}

                        for column in columns:
                            col_vals = np.asarray(model_df[column].values)
                            nan_idx = np.isnan(col_vals)
                            results[column+"-Spearman"] = stats.spearmanr(acc_diff[~nan_idx], col_vals[~nan_idx]).correlation
                            results[column+"-Kendall"] = stats.kendalltau(acc_diff[~nan_idx], col_vals[~nan_idx]).correlation

                        output = output.append(results, ignore_index=True)
                else:
                    model_df = df[df["Model A"] == model]
                    model_df = model_df[~model_df["Model Reduced B"].isin(["deit"])]
                    
                    if "0.9" in model or "1.0" in model:
                        ratios = ["0.9", "1.0"]
                    elif "0.7" in model:
                        ratios = ["0.7"]
                    elif "0.5" in model:
                        ratios = ["0.5"]
                    elif "0.25" in model:
                        ratios = ["0.25"]

                    model_df = model_df[model_df["Ratio B"].isin(ratios)]
                    acc_diff = np.asarray(model_df["Acc Diff"].values)
                    results = {"Model": model}

                    for column in columns:
                        col_vals = np.asarray(model_df[column].values)
                        nan_idx = np.isnan(col_vals)
                        results[column+"-Spearman"] = stats.spearmanr(acc_diff[~nan_idx], col_vals[~nan_idx]).correlation
                        results[column+"-Kendall"] = stats.kendalltau(acc_diff[~nan_idx], col_vals[~nan_idx]).correlation
                    
                    output = output.append(results, ignore_index=True)
                
            output.to_csv(os.path.join(output_dir, f"correlations_comparison_models_{capacity}_{dataset}.csv"), sep=";", index=False)
            print(output)

    if __name__ == '__main__':
        correlate_capacity_data()