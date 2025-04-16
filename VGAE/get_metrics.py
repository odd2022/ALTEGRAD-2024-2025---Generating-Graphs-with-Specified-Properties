

import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import community as community_louvain
import torch
from torch.utils.data import Subset


def extract_metrics_from_graphs(graphs):
    metrics = []
    for G in graphs:
        metrics_to_add = []
        metrics_to_add.append(G.number_of_nodes())
        metrics_to_add.append(G.number_of_edges())
        metrics_to_add.append(sum(dict(G.degree()).values()) / G.number_of_nodes())
        metrics_to_add.append(sum(nx.triangles(G).values()) // 3)
        metrics_to_add.append(nx.transitivity(G))
        metrics_to_add.append(max(nx.core_number(G).values()))
        metrics_to_add.append(len(nx.community.louvain_communities(G, seed=123)))
        metrics.append(metrics_to_add)
    return metrics

def extract_metrics_from_agency_matrix(adj_matrices):
    metrics = []
    graphs = []
    for adj_matrix in adj_matrices:
        adj_matrix = adj_matrix.detach().numpy()
        G = nx.from_numpy_array(adj_matrix)
        graphs.append(G)
    metrics = extract_metrics_from_graphs(graphs)
    return metrics



def compare_metrics(metrics1, metrics2):
    true_stat_normalized1 = torch.where(metrics1 != 0, torch.ones_like(metrics1), torch.zeros_like(metrics1))
    pred_stat_normalized1 = torch.where(metrics1 != 0, metrics2 / metrics1, torch.zeros_like(metrics2))
    # pred_stat_normalized1 = torch.where(metrics1 != 0, metrics2 / metrics1, metrics2)

    mae_normalized1 = torch.sum(torch.mean(torch.abs(true_stat_normalized1 - pred_stat_normalized1), dim=1))
    MAE_weighted = mae_normalized1/1000
    return MAE_weighted
