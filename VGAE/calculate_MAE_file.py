import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
import re
from torch_geometric.loader import DataLoader
import community as community_louvain
from utils import preprocess_dataset
from get_metrics import extract_metrics_from_graphs, extract_metrics_from_agency_matrix, compare_metrics
import argparse
import json



def calculate_metrics(df1):
    all_stats = []
    dataset = preprocess_dataset("test", 50, 10)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    for k, data in enumerate(tqdm(loader, desc='Processing test set')):
        data = data.to(device)
        stat = data.stats
        stat = torch.tensor(stat, dtype=torch.float32)
        stat_d = torch.reshape(stat, (-1, 7))
        for i in range(stat.size(0)):
            stat_x = stat_d[i]
            all_stats.append(stat_x)
    all_stats_matrix = torch.stack(all_stats)

    sum = 0
    list_grahs = []
    for index, row in df1.iterrows():
        graph_id = row["graph_id"]
        edge_list = row["edge_list"]
        pattern = r"\((\d+),\s*(\d+)\)"
        matches1 = re.findall(pattern, edge_list)
        edge_list_df1 = [(int(a), int(b)) for a, b in matches1]
        # graph from edge_list_df1
        G = nx.Graph()
        G.add_edges_from(edge_list_df1)
        list_grahs.append(G)
    metrics_output = extract_metrics_from_graphs(list_grahs)
    # to tensor
    metrics_output = torch.tensor(metrics_output, dtype=torch.float32)
    # print(metrics_output.shape)
    MAE_computed = compare_metrics(all_stats_matrix, metrics_output)

    print("MAE ", MAE_computed)

    return MAE_computed



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



name_exp = "outputs/experiment_tau_2_2025-01-15_19-10-24" # change with the name of the experiment
df = pd.read_csv(f"{name_exp}/output.csv")
# df = pd.read_csv(f"{name_exp}/output_random_test_100.csv")
calculate_metrics(df)