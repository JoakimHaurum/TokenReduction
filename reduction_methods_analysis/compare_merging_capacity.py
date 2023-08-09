
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.metrics import homogeneity_completeness_v_measure

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

from reduction_methods_analysis.analysis_utils import get_model_pair, map_cluster_centers

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

    compute_iou = args.iou
    use_distill = args.use_distill

    overview_df = pd.read_csv(overview_path, sep=",")
    
    overview_df["Distilled Model"] = overview_df.apply(lambda r: (True if "Distill" in r["Tags"] else False), axis=1)
    overview_df = overview_df[overview_df["Distilled Model"] == use_distill]
    
    overview_df["Model Capacity"] = overview_df.apply(lambda r: (r["model"].split("_")[1]), axis=1)
    overview_df["Model Reduced"] = overview_df.apply(lambda r: ("linf" + "-" + r["model"] ) if "linf" in r["model"] else r["model"], axis=1)
    overview_df["Model Reduced"] = overview_df.apply(lambda r: r["Model Reduced"].split("_")[0], axis=1)
    overview_df["model"] = overview_df.apply(lambda r: (r["model"] + "-" + r["heuristic_pattern"]) if "heuristic" in r["model"] else r["model"], axis=1)
    overview_df = overview_df.sort_values(by=["Model Reduced","Model Capacity"], ascending=[True, False])


    print(overview_df)


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

        if "deit" in model1:
            continue

        ratio1 = row1["keep_rate"].replace("[","").replace("]","")
        if ratio1 != "":
            ratio1 = float(ratio1)
        loc1 = row1["reduction_loc"].replace("[","").replace("]","")
        loc1 = [int(idx) for idx in loc1.split(",")]

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

        assign_maps = True if "Assignment_Maps" in stage_tasks else False

        if not assign_maps:
            continue

        model1_name = get_model_pair(model1, loc1, ratio1)
        print(model1_name, stages)
        
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

            if "deit" in model2:
                continue

            ratio2 = row2["keep_rate"].replace("[","").replace("]","")
            if ratio2 != "":
                ratio2 = float(ratio2)
            loc2 = row2["reduction_loc"].replace("[","").replace("]","")
            loc2 = [int(idx) for idx in loc2.split(",")]

            if ratio1 != ratio2:
                continue

            if loc1 != loc2:
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
            
            assign_maps2 = True if "Assignment_Maps" in stage_tasks2 else False
            if not assign_maps2:
                continue                          

            stage_homogeneity = {key:[] for key in stages if key in stages2}
            stage_completeness = {key:[] for key in stages if key in stages2}
            stage_nmi = {key:[] for key in stages if key in stages2}

            if compute_iou:
                stage_ious_model1 = {key:[] for key in stages if key in stages2}
                stage_ious_model2 = {key:[] for key in stages if key in stages2}

            print("\t{} {}".format(model2_name, stages2))

            for img in keys_list:
                img_dict1 = data1[img]
                img_dict2 = data2[img]
                cluster1 = None
                cluster2 = None
                for stage in stages:
                    if stage in stages2:
                        cluster1_stage = img_dict1[stage]["Assignment_Maps"]
                        cluster2_stage = img_dict2[stage]["Assignment_Maps"]
                        cluster1, cluster2 = map_cluster_centers(cluster1_stage, cluster2_stage, cluster1, cluster2, first_stage = stage==stages[0])

                        homogeneity, completeness, nmi = homogeneity_completeness_v_measure(cluster1, cluster2)
                        stage_homogeneity[stage].append(homogeneity)
                        stage_completeness[stage].append(completeness)
                        stage_nmi[stage].append(nmi)

            res_dict = {"Model A": model1_name,
                        "Model B": model2_name,
                        "Ratio": ratio1,
                        "Loc": loc1,
                        **{"Homogeneity-{}-Mean".format(stage): np.mean(stage_homogeneity[stage]) for stage in stages if stage in stages2},
                        **{"Homogeneity-{}-std".format(stage): np.std(stage_homogeneity[stage]) for stage in stages if stage in stages2},
                        **{"Completeness-{}-Mean".format(stage): np.mean(stage_completeness[stage]) for stage in stages if stage in stages2},
                        **{"Completeness-{}-std".format(stage): np.std(stage_completeness[stage]) for stage in stages if stage in stages2},
                        **{"NMI-{}-Mean".format(stage): np.mean(stage_nmi[stage]) for stage in stages if stage in stages2},
                        **{"NMI-{}-std".format(stage): np.std(stage_nmi[stage]) for stage in stages if stage in stages2}}
            
            output = output.append(res_dict, ignore_index=True)
        output.to_csv(os.path.join(output_dir, output_file), sep=";", index=False)



if __name__ == '__main__':
    compare_main()
