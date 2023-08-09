import pandas as pd 
from datetime import datetime
import wandb
import csv
import argparse
import os
api = wandb.Api()

parser = argparse.ArgumentParser('Get Weights and Biases results', add_help=False)
parser.add_argument('--entity', default="", type=str)
parser.add_argument('--project', default="", type=str)
parser.add_argument('--output_path', default="", type=str)
args = parser.parse_args()

entity = args.entity
project = args.project
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

# Project is specified by <entity/project-name>
runs = api.runs("{}/{}{}".format(entity, project))

date_now = datetime.now()
date_format = "%y%m%d"
date = date_now.strftime(date_format)

runs_df = pd.DataFrame()
for run in runs: 

    if "Not Best" in run.tags or "hidden" in run.tags:
        continue
    
    config_file = run.config
    summary_file = run.summary._json_dict

    if run.state == "running":
        continue

    if "max_accuracy" not in summary_file.keys():
        continue

    if not "heuristic_pattern" in config_file.keys():
        config_file["heuristic_pattern"] = ""

    print(run.name, run.tags)
    
    if "epoch" in summary_file.keys():
        epoch = summary_file["epoch"]
    else:
        epoch = 0

    data_dict = {"Name": run.name,
                 "max_accuracy": summary_file["max_accuracy"],
                 "epoch": epoch,
                 "epochs": config_file["epochs"],
                 "keep_rate": str(config_file["keep_rate"]),
                 "reduction_loc": str(config_file["reduction_loc"]),
                 "heuristic_pattern": config_file["heuristic_pattern"],
                 "model": config_file["model"],
                 "output_dir": config_file["output_dir"],
                 "Tags": run.tags,
                 "Created": run.created_at
                 }
    df_dictionary = pd.DataFrame([data_dict])
    runs_df = pd.concat([runs_df, df_dictionary], ignore_index=True)

runs_df = runs_df.sort_values(by=["Created"])

runs_df.to_csv(os.path.join(output_path, "{}_{}_{}.csv".format(project, str(date))), index=False, quoting=csv.QUOTE_ALL)