import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast
import json

from torch.utils.tensorboard import SummaryWriter

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset
from get_metrics import extract_metrics_from_graphs, extract_metrics_from_agency_matrix, compare_metrics


from torch.utils.data import Subset
np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Name of the run
parser.add_argument('--run-name', type=str, default="experiment", help="Name of the run for logging")

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=3000, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=32, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=2, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100000, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=600, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=True, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

# Encoding of the prompt
parser.add_argument('--prompt-encoding', type=str, default="extract_numbers", help="Prompt encoding method (default: extract_numbers)") # could be "extract_numbers" "encode_prompt", "combine_feats"

# Graph embedding
parser.add_argument('--graph-embedding', type=str, default="basic", help="Graph embedding method (default: 'basic')") # could also be "DW" or "BERT"

parser.add_argument('--tau', type=float, default=1, help="Tau value for the decoder (default: 1)")

parser.add_argument('--reduction-tau', type=float, default=1, help="Reduction factor for tau (default: 1)")
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


name_exp = 'experiment_tau_2_2025-01-15_19-10-24'

train_autoencoder = args.train_autoencoder
train_denoiser = args.train_denoiser
tau = args.tau

# Tensorboard writer
if args.train_autoencoder:
    run_name = args.run_name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f'runs/{run_name}_tau_{tau}_{current_time}'
    output_dir = f'outputs/{run_name}_tau_{tau}_{current_time}'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    
else:
    run_dir = f'runs/{name_exp}'
    output_dir = f'outputs/{name_exp}'
    with open(f'{output_dir}/args.json', 'r') as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.train_autoencoder = train_autoencoder
    args.train_denoiser = train_denoiser


# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, feature_extractor=args.prompt_encoding, graph_embedding=args.graph_embedding)

validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, feature_extractor=args.prompt_encoding, graph_embedding=args.graph_embedding)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, feature_extractor=args.prompt_encoding)

# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# initialize VGAE model
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes, dropout=args.dropout).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)



writer = SummaryWriter(log_dir=run_dir)


training_autoencoder_time = 60 * 60 * 2 # 2 hours maximum for training autoencoder
start_time = datetime.now()
# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    best_val_bce_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        train_loss_all_bce = 0
        cnt_train=0

        for data in train_loader:
            # data = torch.tensor(data).to(device)
            data = data.to(device)
            optimizer.zero_grad()
            # print(data.stats.shape)
            loss, recon, kld, sim_prop, bce  = autoencoder.loss_function(data, tau=tau)
            # loss, recon, kld, sim_prop, bce  = autoencoder.combined_loss_with_targets(data, tau=tau)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            train_loss_all_bce += bce.item()
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()
        writer.add_scalar('autoencoder/Loss/Train', train_loss_all / cnt_train, epoch)
        writer.add_scalar('autoencoder/Reconstruction Loss/Train', train_loss_all_recon / cnt_train, epoch)
        writer.add_scalar('autoencoder/BCE Loss/Train', train_loss_all_bce / cnt_train, epoch)
        writer.add_scalar('autoencoder/KLD Loss/Train', train_loss_all_kld / cnt_train, epoch)
        writer.add_scalar('autoencoder/Similarity Property/Train', sim_prop, epoch)

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_bce = 0
        val_loss_all_kld = 0
        if epoch % 5 == 0:
            val_loss_prop = 0

        for data in val_loader:
            data = data.to(device)
            loss, recon, kld, sim_prop, bce  = autoencoder.loss_function(data, tau=tau)
            # loss, recon, kld, sim_prop, bce  = autoencoder.combined_loss_with_targets(data, tau=tau)
            val_loss_all_recon += recon.item()
            val_loss_all_kld += kld.item()
            val_loss_all_bce += bce.item()
            val_loss_all += loss.item()
            cnt_val+=1
            val_count += torch.max(data.batch)+1

        
        writer.add_scalar('autoencoder/Loss/Validation', val_loss_all / cnt_val, epoch)
        writer.add_scalar('autoencoder/Reconstruction Loss/Validation', val_loss_all_recon / cnt_val, epoch)
        writer.add_scalar('autoencoder/BCE Loss/Validation', val_loss_all_bce / cnt_val, epoch)
        writer.add_scalar('autoencoder/KLD Loss/Validation', val_loss_all_kld / cnt_val, epoch)
        writer.add_scalar('autoencoder/Similarity Property/Validation', sim_prop, epoch)


        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Train Prop loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}, Val Prop loss: {:.2f}'.format(dt_t, epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, sim_prop, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val, sim_prop))
        scheduler.step()
        if epoch % 100 == 0:
            tau = max(tau * args.reduction_tau, 0.01)
        

        if best_val_bce_loss >= val_loss_all_bce:
            best_val_bce_loss = best_val_bce_loss
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, f'{output_dir}/autoencoder.pth.tar')
        # if best_val_loss >= val_loss_all:
        #     best_val_loss = val_loss_all
        #     torch.save({
        #         'state_dict': autoencoder.state_dict(),
        #         'optimizer' : optimizer.state_dict(),
        #     }, f'{output_dir}/autoencoder.pth.tar')
        if (datetime.now() - start_time).seconds > training_autoencoder_time:
            break
else:
    checkpoint = torch.load(f'{output_dir}/autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])
    autoencoder.eval()
    val_loss_all = 0
    val_count = 0
    cnt_val = 0
    val_loss_all_recon = 0
    val_loss_all_bce = 0
    val_loss_all_kld = 0
    val_loss_rand_all = 0
    for data in val_loader:
        data = data.to(device)
        loss, recon, kld, sim_prop, bce  = autoencoder.loss_function(data, tau=tau)
        # loss, recon, kld, sim_prop, bce  = autoencoder.combined_loss_with_targets(data)
        val_loss_all_recon += recon.item()
        val_loss_all_kld += kld.item()
        val_loss_all_bce += bce.item()
        val_loss_all += loss.item()
        cnt_val+=1
        val_count += torch.max(data.batch)+1

        stat = data.stats
        # to tensor
        stat = torch.tensor(stat).to(device)
        bs = stat.size(0)

        x_g1  = autoencoder.encoder(data)
        mu = autoencoder.fc_mu(x_g1)
        logvar = autoencoder.fc_logvar(x_g1)
        x_g2 = autoencoder.reparameterize(mu, logvar)
        adj = autoencoder.decoder(x_g2, data, tau=tau)
        Gs_generated = []
        for i in range(stat.size(0)):
            Gs_generated.append(construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy()))
        metrics_graphs = extract_metrics_from_graphs(Gs_generated)
        metrics_graphs = torch.tensor(metrics_graphs).to(device)
        val_loss_rand_all += compare_metrics(stat, metrics_graphs)
    
    print('autoencoder Loss: {:.5f}, autoencoder Reconstruction Loss: {:.2f}, autoencoder KLD Loss: {:.2f}, autoencoder Prop loss: {:.2f}, MAE: {:.5f}'.format(val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val, sim_prop, val_loss_rand_all))

autoencoder.eval()


# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
if args.prompt_encoding == "extract_numbers":
    n_cond = 7
elif args.prompt_encoding == "encode_prompt":
    n_cond = 768
else:
    n_cond = 768 + 7
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=n_cond, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

train_denoiser_time = 60* 60 * 2 # 4 hours
start_time = datetime.now()
# Train denoising model
if args.train_denoiser: 
    best_val_loss = np.inf
    best_val_rand_loss = np.inf
    best_test_rand_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long() # random diffusion step for each graph
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()
        writer.add_scalar('denoiser/Loss/Train', train_loss_all / train_count, epoch)

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        val_loss_rand_all = 0
        test_loss_rand_all = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)
            if epoch % 20 == 0:
                stat = data.stats
                # to tensor
                stat = torch.tensor(stat).to(device)
                bs = stat.size(0)

                samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
                x_sample = samples[-1]
                adj = autoencoder.decode_mu(x_sample, data=data, tau=tau)
                stat_d = torch.reshape(stat, (-1, n_cond))
                Gs_generated = []
                for i in range(stat.size(0)):
                    stat_x = stat_d[i]
                    Gs_generated.append(construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy()))
                metrics_graphs = extract_metrics_from_graphs(Gs_generated)
                metrics_graphs = torch.tensor(metrics_graphs).to(device)
                val_loss_rand_all += compare_metrics(stat, metrics_graphs)
                writer.add_scalar('denoiser/MAE/Val', val_loss_rand_all, epoch)
                if best_val_rand_loss >= val_loss_rand_all:
                    best_test_rand_loss = val_loss_rand_all
                    torch.save({
                        'state_dict': denoise_model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, f'{output_dir}/denoise_model_best_val_rand.pth.tar')
                
        
        if epoch % 20 == 0:
            for data in test_loader:
                data = data.to(device)
                stat = data.stats
                stat = torch.tensor(stat).to(device)
                bs = stat.size(0)

                samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
                x_sample = samples[-1]
                adj = autoencoder.decode_mu(x_sample, data=data, tau=tau)
                stat_d = torch.reshape(stat, (-1, n_cond))
                Gs_generated = []
                for i in range(stat.size(0)):
                    stat_x = stat_d[i]
                    Gs_generated.append(construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy()))
                metrics_graphs = extract_metrics_from_graphs(Gs_generated)
                metrics_graphs = torch.tensor(metrics_graphs).to(device)
                test_loss_rand_all += compare_metrics(stat, metrics_graphs)
            
            writer.add_scalar('denoiser/MAE/Test', test_loss_rand_all, epoch)
            if best_test_rand_loss >= test_loss_rand_all:
                best_test_rand_loss = test_loss_rand_all
                torch.save({
                    'state_dict': denoise_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, f'{output_dir}/denoise_model_best_test_rand.pth.tar')
        writer.add_scalar('denoiser/Loss/Validation', val_loss_all / val_count, epoch)
        

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, f'{output_dir}/denoise_model_best_val.pth.tar')
            
        if (datetime.now() - start_time).seconds > train_denoiser_time:
            break
else:
    # checkpoint = torch.load(f'{output_dir}/denoise_model.pth.tar')
    checkpoint = torch.load(f'{output_dir}/denoise_model_best_val_rand.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])


writer.close()

denoise_model.eval()

del train_loader, val_loader


# Save to a CSV file
with open(f"{output_dir}/output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        
        stat = data.stats
        stat = torch.tensor(stat).to(device)
        bs = stat.size(0)

        graph_ids = data.filename

        samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample, data=data, tau=tau)
        stat_d = torch.reshape(stat, (-1, args.n_condition))


        for i in range(stat.size(0)):
            stat_x = stat_d[i]

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            stat_x = stat_x.detach().cpu().numpy()

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])


test_loss_rand_all = 0
# Save to a CSV file
with open(f"{output_dir}/output_random_test_100.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        
        stat = data.stats
        stat = torch.tensor(stat).to(device)
        bs = stat.size(0)

        graph_ids = data.filename
        list_adj = []
        for _ in range(100):
            samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
            x_sample = samples[-1]
            adj = autoencoder.decode_mu(x_sample, data=data, tau=tau)
            list_adj.append(adj)
        stat_d = torch.reshape(stat, (-1, n_cond))

        list_Gs_generated = []
        for i in range(stat.size(0)):
            stat_x = stat_d[i]
            stat_x = stat_x.view(-1, 7)
            stat_x = torch.tensor(stat_x).to(device)
            best_mae = np.inf
            best_Gs_generated = None
            first_Gs_generated = construct_nx_from_adj(list_adj[0][i,:,:].detach().cpu().numpy())
            for adj in list_adj:
                Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
                metrics_graph = extract_metrics_from_graphs([Gs_generated])
                metrics_graph = torch.tensor(metrics_graph).to(device)
                metrics_graph = metrics_graph.view(-1, 7)
                mae = compare_metrics(stat_x, metrics_graph)
                if mae < best_mae:
                    best_mae = mae
                    best_Gs_generated = Gs_generated
            
            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in best_Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])
            # list_Gs_generated.append(Gs_generated)
            list_Gs_generated.append(best_Gs_generated)

        metrics_graphs = extract_metrics_from_graphs(list_Gs_generated)
        #to tensor
        metrics_graphs = torch.tensor(metrics_graphs).to(device)
        test_loss_rand_all += compare_metrics(stat, metrics_graphs)
print("MAE", test_loss_rand_all)