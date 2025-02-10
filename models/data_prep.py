# -*- coding: utf-8 -*-
import pandas as pd
import yaml 
from tqdm import tqdm 
import pickle

# Get config values
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract relevant datasets
print(f"Extracting datasets from {config['base_path']}")
print("List of file names available")
for fn in config['file_name']:
    print(fn)

# Getting list of all available unique individual IDs
uid = []
for fn in config['file_name']:
    df = pd.read_excel(config['base_path']+fn, engine='openpyxl')
    uid.extend(list(df[config['id_label']].unique()))
uid = list(set(uid))

# Creating a dictionary per ID with all information available in input files
pop_dict = {}
for fn in config['file_name']:
    print(f"Processing data in {fn}")
    df = pd.read_excel(config['base_path']+fn, engine='openpyxl')
    
    for i in tqdm(uid):
        if i in pop_dict.keys():
            # Key is the file name and ID
            pop_dict[i].update({fn.split('.')[0] : df[df[config['id_label']] == i].to_dict('records')})
        else:
            pop_dict[i] = {fn.split('.')[0] : df[df[config['id_label']] == i].to_dict('records')}

with open(config['pop_dict'], 'wb') as file:
    pickle.dump(pop_dict, file)