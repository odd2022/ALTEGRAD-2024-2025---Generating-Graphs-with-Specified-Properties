from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch_geometric.data import Data
import os
import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from datetime import datetime
from utils.extract_feats import extract_numbers_feats, extract_numbers, encode_prompt, combine_feats, extract_feats
import csv


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f'outputs/experiment_knn_{current_time}'
os.makedirs(run_dir, exist_ok=True)

def preprocess_dataset(dataset):
    graph_path = 'data/'+dataset+'/graph'
    desc_path = 'data/'+dataset+'/description'
    files = [f for f in os.listdir(graph_path)]
    count = 0
    dataset_feats = {"feats": [], "edges": [], "graph_ids": []}   
    for fileread in tqdm(files):
        dict_graph = {}
        tokens = fileread.split("/")
        idx = tokens[-1].find(".")
        filen = tokens[-1][:idx]
        extension = tokens[-1][idx+1:]
        fread = os.path.join(graph_path,fileread)
        fstats = os.path.join(desc_path,filen+".txt")
        file = open(fstats, "r")
        file = open(fstats, "r")
        desc = file.read()
        feats = extract_feats(desc, feature_extractor="extract_numbers", file=fstats)
        file.close()
        dataset_feats["feats"] += [feats]
        
        if extension=="graphml":
            G = nx.read_graphml(fread)
            # Convert node labels back to tuples since GraphML stores them as strings
            G = nx.convert_node_labels_to_integers(
                G, ordering="sorted"
            )
        else:
            G = nx.read_edgelist(fread)
        str_edge_list = ", ".join([f"({u}, {v})" for u, v in G.edges()])

        dataset_feats["edges"] += [str_edge_list]
        dataset_feats["graph_ids"] += [filen]
    
    return dataset_feats

valid_data = preprocess_dataset("valid")
train_data = preprocess_dataset("train")

def preprocess_testset():
    data_lst = {"feats": [], "graph_ids": []}
    desc_file = 'data/test/test.txt'

    fr = open(desc_file, "r")
    for line in fr:
        line = line.strip()
        tokens = line.split(",")
        graph_id = tokens[0]
        desc = tokens[1:]
        desc = "".join(desc)
        prop = extract_feats(desc, feature_extractor="extract_numbers")
        data_lst["feats"] += [prop]
        data_lst["graph_ids"] += [graph_id]
    fr.close()                    

    return data_lst

data_lst = preprocess_testset()

# predict edges for valid_data taking the closest graph in train_data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

X_train = np.array(train_data["feats"])
X_valid = np.array(valid_data["feats"])
X_test = np.array(data_lst["feats"])
knn = NearestNeighbors(n_neighbors=1)
knn.fit(X_train)
distances_valid, indices_valid = knn.kneighbors(X_valid)



with open(f'outputs/experiment_knn_{current_time}/output_valid.csv', 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    count = 0
    for i, (idx, dist) in enumerate(zip(indices_valid, distances_valid)):
        graph_id = valid_data['graph_ids'][i]
        edge_list = train_data['edges'][i]
        writer.writerow([graph_id, edge_list])
        

distances_test, indices_test = knn.kneighbors(X_test)

with open(f'outputs/experiment_knn_{current_time}/output_test.csv', 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    count = 0
    for i, (idx, dist) in enumerate(zip(indices_test, distances_test)):
        graph_id = data_lst['graph_ids'][i]
        edge_list = train_data['edges'][i]
        writer.writerow([graph_id, edge_list])

