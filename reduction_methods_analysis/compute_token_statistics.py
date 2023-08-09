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

from reduction_methods_analysis.analysis_utils import get_model_pair

parser = argparse.ArgumentParser(description='Compute average reduction pattern for reduction method(s)')
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
    overview_df["model"] = overview_df.apply(lambda r: (r["model"] + "-" + r["heuristic_pattern"]) if "heuristic" in r["model"] else r["model"], axis=1)
    overview_df = overview_df.sort_values(by=["model","keep_rate"], ascending=[True, False])    

    print(overview_df)

    max_stages = 0

    model_dict = {}
    for index, row in overview_df.iterrows():
        name1 = row["Name"]
        model1 = row["model"]
        
        if "deit" not in model1:
            ratio1 = row["keep_rate"].replace("[","").replace("]","")
            if ratio1 != "":
                ratio1 = float(ratio1)
            loc1 = row["reduction_loc"].replace("[","").replace("]","")
            loc1 = [int(idx) for idx in loc1.split(",")]
        else:
            ratio1 = ""
            loc1 = ""

        file1 = name1 + "_viz_results.json"
        path1 = os.path.join(model_parent_dir, file1)

        if not os.path.isfile(path1):
            continue

        with open(path1) as json_file:
            data1 = json.load(json_file)

        keys_list = list(data1.keys())

        acc1 = data1["Top1-Acc"]
        acc5 = data1["Top5-Acc"]

        keys_list.remove("Model")
        keys_list.remove("Ratio")
        keys_list.remove("Location")
        keys_list.remove("Top1-Acc")
        keys_list.remove("Top5-Acc")
        keys_list.remove("Params")

        if "deit" in model1:
            stages  = [x for x in range(12)]
            stage_tasks = []
        else:
            stages = [x for x in list(data1[keys_list[0]].keys()) if "Stage" in x]
            stage_tasks = list(data1[keys_list[0]][stages[0]])

        if len(stages) > max_stages:
            max_stages = len(stages)

        kept_tokens = True if "Kept_Token" in stage_tasks else False
        assign_maps = True if "Assignment_Maps" in stage_tasks else False

        model1_name = get_model_pair(model1, loc1, ratio1)
        print(model1_name, stages)


        max_depth = 12
        base_patch_count = 14*14

        if kept_tokens or assign_maps:
            stage_reduction_rate = {stage: [] for stage in stages}
        if kept_tokens:
            stage_token_depth = [[] for _ in range(base_patch_count)]

        if "deit" in model1:
            stage_reduction_rate = {stage: [1.] for stage in stages}
            stage_token_depth = [[max_depth] for _ in range(base_patch_count)]
        else:
            for img in keys_list:
                img_dict = data1[img]
                prev_set = set([idx for idx in range(base_patch_count)])
                
                for stage in stages:
                    if kept_tokens:
                        token = set(img_dict[stage]["Kept_Token"])
                        token.discard(-1)
                    elif assign_maps:
                        token = set(img_dict[stage]["Assignment_Maps"])

                    difference = list(prev_set.difference(token))
                    prev_set = token

                    stage_reduction_rate[stage].append(len(token)/base_patch_count)
                    
                    if kept_tokens:
                        depth = int(stage[6:])
                        for idx in difference:
                            if idx >= base_patch_count:
                                continue
                            
                            stage_token_depth[idx].append(depth)
                if kept_tokens:
                    for idx in prev_set:
                        if idx >= base_patch_count:
                            continue
                        
                        stage_token_depth[idx].append(max_depth)

        if kept_tokens or "deit" in model1:
            mean_token_depth_spatial = [0]*base_patch_count
            std_token_depth_spatial = [0]*base_patch_count

            global_token_depth_list = []
            
            for idx in range(base_patch_count):
                mean_token_depth_spatial[idx] = np.mean(stage_token_depth[idx])
                std_token_depth_spatial[idx] = np.std(stage_token_depth[idx])
                global_token_depth_list.extend(stage_token_depth[idx])
            
            mean_token_depth = np.mean(global_token_depth_list)
            std_token_depth = np.std(global_token_depth_list)
            
        mean_reduction_rate = [0]*len(stages)
        std_reduction_rate = [0]*len(stages)

        for idx, stage in enumerate(stages):            
            mean_reduction_rate[idx] = np.mean(stage_reduction_rate[stage])
            std_reduction_rate[idx] = np.std(stage_reduction_rate[stage])
        
    
        res_dict = {"Model": model1,
                    "Acc-Top1": acc1,
                    "Acc-Top5": acc5,
                    "Ratio": ratio1,
                    "Loc": loc1,
                    "Stages": stages,
                    "Mean-Reduction": mean_reduction_rate,
                    "Std-Reduction": std_reduction_rate
        }

        if kept_tokens or "deit" in model1:
            res_dict = {**res_dict,
                        "Mean-Token-Depth": mean_token_depth,
                        "Std-Token-Depth": std_token_depth,    
                        "Mean-Token-Depth-Spatial": mean_token_depth_spatial,
                        "Std-Token-Depth-Spatial": std_token_depth_spatial}

        model_dict[model1_name] = res_dict

        with open(os.path.join(output_dir, output_file), "w") as outfile:
            json.dump(model_dict, outfile, indent=4)


if __name__ == '__main__':
    compare_main()
