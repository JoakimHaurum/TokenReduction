import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser(description='Correlated data computed across keep rates for the same reduction method(s) and difference in metric performance')
parser.add_argument('--parent_dir', default="", type=str)
parser.add_argument("--datasets", nargs='+', type=str, default = ["IM","NAB","COCO","NUS"])
parser.add_argument("--output_dir", default="", type=str)

def correlate_rates_data():
    args = parser.parse_args()
    base_path = args.parent_dir
    datasets = args.datasets
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        path = f"collated_comparison_rates_{dataset}.csv"

        df = pd.read_csv(os.path.join(base_path, path), sep=";")
        df["Model Reduced"] = df.apply(lambda r: (r["Model A"][:-5]), axis=1)

        columns = list(df.columns)
        columns.remove("Model A")
        columns.remove("Model B")
        columns.remove("Ratio A")
        columns.remove("Ratio B")
        columns.remove("Acc A")
        columns.remove("Acc B")
        columns.remove("Acc Diff")
        columns.remove("Model Reduced")

        unique_models = df["Model Reduced"].unique()
        output = pd.DataFrame()     

        for model in unique_models:
            model_df = df[df["Model Reduced"] == model]        
            acc_diff = np.asarray(model_df["Acc Diff"].values)
            results = {"Model": model}

            for column in columns:
                    col_vals = np.asarray(model_df[column].values)
                    nan_idx = np.isnan(col_vals)
                    results[column+"-Spearman"] = stats.spearmanr(acc_diff[~nan_idx], col_vals[~nan_idx]).correlation
                    results[column+"-Kendall"] = stats.kendalltau(acc_diff[~nan_idx], col_vals[~nan_idx]).correlation

            output = output.append(results, ignore_index=True)
            
        output.to_csv(os.path.join(output_dir, f"correlations_comparison_rates_{dataset}.csv"), sep=";", index=False)
        print(output)

if __name__ == '__main__':
    correlate_rates_data()