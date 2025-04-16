"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt
import pandas as pd
import ast
from tqdm import tqdm
# Loads the karate network
data_path = "data/dataframe/train.csv"
# data_path = "/data/dataframe/valid.csv"
df = pd.read_csv(data_path)

n_max_nodes = 50
ls_embeddings = np.zeros((len(df), n_max_nodes, 128))

for index, row in tqdm(df.iterrows()):
    graph_id = row['graph_ids']
    edge_list = row['edges'] # not weighted
    # to list
    edge_list = ast.literal_eval(edge_list)
    G = nx.from_edgelist(edge_list)
    n_dim = 128
    n_walks = 10
    walk_length = 20
    model = deepwalk(G, n_walks, walk_length, n_dim)
    for node in G.nodes():
        embeddings = model.wv[str(node)]
        ls_embeddings[index, node] = embeddings

# save numpy array
output_path = "data/dataframe/train_deepwalk.npy"
# output_path = "data/dataframe/valid_deepwalk.npy"

np.save(output_path, ls_embeddings)
