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

parser = argparse.ArgumentParser(description='Compare pruning-based method(s) across model capacity')
parser.add_argument('--parent_dir', default="", type=str)
parser.add_argument("--dataset_csv", default="", type=str)
parser.add_argument("--capacity", nargs='+', type=str, default = ["small"])
parser.add_argument("--output_file", default="", type=str)
parser.add_argument("--output_dir", default="", type=str)

def compare_main():
    args = parser.parse_args()
    model_parent_dir = args.parent_dir
    overview_path = args.dataset_csv
    output_file = args.output_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_capacities = args.capacity

    overview_df = pd.read_csv(overview_path, sep=",")
    overview_df["Model Capacity"] = overview_df.apply(lambda r: (r["model"].split("_")[1]), axis=1)
    overview_df = overview_df[overview_df["Model Capacity"].isin(model_capacities)]
    overview_df["model"] = overview_df.apply(lambda r: (r["model"] + "-" + r["heuristic_pattern"]) if "heuristic" in r["model"] else r["model"], axis=1)

    print(overview_df)

    output = pd.DataFrame()
    model_pairs = []
    file_pairs = []
    for index1, row1 in overview_df.iterrows():
        name1 = row1["Name"]
        model1 = row1["model"]

        valid_capacity1 = False
        for capacity in model_capacities:
            if capacity in model1:
                valid_capacity1 = True
        
        if not valid_capacity1:
            continue
        
        if "deit" in model1:
            continue

        ratio1 = row1["keep_rate"].replace("[","").replace("]","")
        if ratio1 != "":
            ratio1 = float(ratio1)
        loc1 = row1["reduction_loc"].replace("[","").replace("]","")
        loc1 = [int(idx) for idx in loc1.split(",")]

        if "heuristic" in model1 or "ats" in model1:
            continuous_model1 = True
        else:
            continuous_model1 = False
        multi_loc1 = len(loc1) != 1

        file1 = name1 + "_viz_results.json"

        path1 = os.path.join(model_parent_dir, file1)

        if not os.path.isfile(path1):
            continue

        with open(path1) as json_file:
            data1 = json.load(json_file)

        keys_list = list(data1.keys())

        keys_list.remove("Model")
        keys_list.remove("Ratio")
        keys_list.remove("Location")
        keys_list.remove("Top1-Acc")
        keys_list.remove("Top5-Acc")
        keys_list.remove("Params")

        stages = [x for x in list(data1[keys_list[0]].keys()) if "Stage" in x]
        stage_tasks = list(data1[keys_list[0]][stages[0]])

        kept_tokens = True if "Kept_Token" in stage_tasks else False

        if not kept_tokens:
            continue

        model1_name = get_model_pair(model1, loc1, ratio1)
        print(model1_name, stages)
        
        for index2, row2 in overview_df.iterrows():   
            name2 = row2["Name"]
            model2 = row2["model"]
       
            valid_capacity2 = False
            for capacity in model_capacities:
                if capacity in model2:
                    valid_capacity2 = True
            
            if not valid_capacity2:
                continue

            if "deit" in model2:
                continue

            ratio2 = row2["keep_rate"].replace("[","").replace("]","")
            if ratio2 != "":
                ratio2 = float(ratio2)
            loc2 = row2["reduction_loc"].replace("[","").replace("]","")
            loc2 = [int(idx) for idx in loc2.split(",")]
            
            
            if "heuristic" in model2 or "ats" in model2:
                continuous_model2 = True
            else:
                continuous_model2 = False
            multi_loc2 = len(loc2) != 1
            
            if model1 == model2:
                continue
            if not continuous_model1 and not continuous_model2:
                if ratio1 != ratio2:
                    continue    
                if loc1 != loc2:
                    continue
            else:
                if multi_loc1 != multi_loc2:
                    continue

            file12 = name1+"+"+name2
            file21 = name2+"+"+name1

            if file12 in file_pairs or file21 in file_pairs:
                continue

            file2 = name2 + "_viz_results.json"

            path2 = os.path.join(model_parent_dir, file2)

            if path1 == path2:
                continue
            
            if not os.path.isfile(path2):
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

            with open(path2) as json_file:
                data2 = json.load(json_file)

            keys_list2 = list(data2.keys())
                
            keys_list2.remove("Model")
            keys_list2.remove("Ratio")
            keys_list2.remove("Location")
            keys_list2.remove("Top1-Acc")
            keys_list2.remove("Top5-Acc")
            keys_list2.remove("Params")

            stages2 = [x for x in list(data2[keys_list2[0]].keys()) if "Stage" in x]
            stage_tasks2 = list(data2[keys_list2[0]][stages2[0]])
            
            kept_tokens2 = True if "Kept_Token" in stage_tasks2 else False
            if not kept_tokens2:
                continue              
            
            stage_ious = {key:[] for key in stages if key in stages2}

            print("\t{} {}".format(model2_name, stages2))

            for img in keys_list:
                img_dict1 = data1[img]
                img_dict2 = data2[img]
                for stage in stages:
                    if stage in stages2:
                        token1 = set(img_dict1[stage]["Kept_Token"])
                        token1.discard(-1)
                        token2 = set(img_dict2[stage]["Kept_Token"])
                        token2.discard(-1)
                        
                        intersection = len(token1.intersection(token2)) 
                        union = len(token1.union(token2))
                        if intersection > 0 and union > 0:
                            iou = intersection/union
                        else:
                            iou = 0
                        stage_ious[stage].append(iou)

            res_dict = {"Model A": model1_name,
                        "Model B": model2_name,
                        "Ratio A": ratio1,
                        "Loc A": loc1,
                        "Ratio B": ratio2,
                        "Loc B": loc2,
                        **{"{}-Mean".format(stage): np.mean(stage_ious[stage]) for stage in stages if stage in stages2},
                        **{"{}-std".format(stage): np.std(stage_ious[stage]) for stage in stages if stage in stages2}}
            res_dict2 = {"Model A": model2_name,
                        "Model B": model1_name,
                        "Ratio A": ratio2,
                        "Loc A": loc2,
                        "Ratio B": ratio1,
                        "Loc B": loc1,
                        **{"{}-Mean".format(stage): np.mean(stage_ious[stage]) for stage in stages if stage in stages2},
                        **{"{}-std".format(stage): np.std(stage_ious[stage]) for stage in stages if stage in stages2}}
            output = output.append(res_dict, ignore_index=True)
            output = output.append(res_dict2, ignore_index=True)

        output.to_csv(os.path.join(output_dir, output_file), sep=";", index=False)

if __name__ == '__main__':
    compare_main()