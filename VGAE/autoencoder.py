import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from get_metrics import extract_metrics_from_agency_matrix

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x, tau=1):
    #     for i in range(self.n_layers-1):
    #         x = self.relu(self.mlp[i](x))
        
    #     x = self.mlp[self.n_layers-1](x)
    #     x = torch.reshape(x, (x.size(0), -1, 2))
    #     x = F.gumbel_softmax(x, tau=tau, hard=True)[:,:,0]

    #     adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
    #     idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
    #     adj[:,idx[0],idx[1]] = x
    #     adj = adj + torch.transpose(adj, 1, 2)
    #     return adj
    
    def forward(self, x, data, tau=1):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=tau, hard=True)[:,:,0]
        # print(x.shape)
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        stats = data.stats
        stats = torch.tensor(stats, dtype=torch.float32)
        for batch_idx, num_nodes in enumerate(stats[:, 0].long()):
            if num_nodes < self.n_nodes:
                adj[batch_idx, num_nodes:, :] = 0
                adj[batch_idx, :, num_nodes:] = 0
        return adj

class Decoder2(nn.Module):
    def __init__(self, latent_dim, n_nodes):
        """
        Initialisation du décodeur.
        
        :param latent_dim: Taille du vecteur latent en entrée
        :param n_nodes: Nombre de nœuds dans le graphe pour calculer la taille de sortie
        """
        super(Decoder2, self).__init__()
        self.n_nodes = n_nodes
        # Taille du vecteur de sortie
        self.output_dim = 2 * n_nodes * (n_nodes - 1) // 2
        
        # Définir les couches du modèle
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, self.output_dim)
        
    def forward(self, x, tau=1):
        """
        Propagation avant du décodeur.
        
        :param x: Entrée du vecteur latent de taille (batch_size, latent_dim)
        :return: Sortie binaire de taille (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=tau, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)

        return adj

class EncoderDW(nn.Module):
    def __init__(self, latent_dim, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64*12*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x = data.embedding_dw

        x = x.view(-1, 1, 50, 128) # Input shape: (batch_size, 1, 50, 128), where 1 is the channel dimension for grayscale images

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        encoded = torch.relu(self.fc4(x))

        return encoded
# encoder for bert embeddings, the inpu is a 128 dim vector
class EncoderBERT(nn.Module):
    def __init__(self, latent_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, data):
        x = data.edge_list_bert_embedding
        x = x.view(-1, 768)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return x



class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out



# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, graph_embedding=None, dropout=0.2):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        if graph_embedding == "BERT":
            self.encoder = EncoderBERT(hidden_dim_enc, dropout=dropout)
        elif graph_embedding == "DW":
            self.encoder = EncoderDW(hidden_dim_enc, dropout=dropout)
        else:
            self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, dropout=dropout)

        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        # self.decoder = Decoder2(latent_dim, n_max_nodes)

    def forward(self, data, tau=1):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, data, tau=tau)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, data=None, tau=1):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g, data=data, tau=tau)
       return adj

    def decode_mu(self, mu, data=None, tau=1):
       adj = self.decoder(mu, data=data, tau=tau)
       return adj

    def loss_function(self, data, beta=0.05, sim_prop=False, tau=1):
        x_g1  = self.encoder(data)
        mu = self.fc_mu(x_g1)
        logvar = self.fc_logvar(x_g1)
        x_g2 = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g2, data, tau=tau)
        
        bce = F.binary_cross_entropy(adj.view(-1), data.A.view(-1), reduction='mean')
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        

        # loss = recon + beta * kld
        loss = bce # + beta * kld

        return loss, recon, kld, sim_prop, bce
    
    def combined_loss_with_targets(self, data, alpha=0.1, beta=0.1, gamma=0.4, delta=0.4, epsilon=0.1, tau=1):
        """
        Fonction de perte combinée qui calcule automatiquement :
        - Reconstruction binaire (BCE)
        - Différence des degrés des graphes
        - Nombre de nœuds actifs
        - Nombre d'arêtes
        - Différence de densité des graphes
        
        Args:
            adj_pred: Tensor prédit de taille (batch_size, n_nodes, n_nodes)
            adj_true: Tensor vrai de taille (batch_size, n_nodes, n_nodes)
            alpha, beta, gamma, delta, epsilon: Coefficients pour pondérer les différents termes de la perte
        
        Returns:
            total_loss: Perte combinée totale
        """

        x_g1  = self.encoder(data)
        mu = self.fc_mu(x_g1)
        logvar = self.fc_logvar(x_g1)
        x_g2 = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g2, tau=tau)
        
        
        # 1. Reconstruction Loss (Binary Cross-Entropy)
        recon_loss = F.binary_cross_entropy(adj.view(-1), data.A.view(-1), reduction='mean')
        

        # Calcul des métriques cibles
        degree_target = data.A.sum(dim=-1)  # Somme des degrés des nœuds (batch_size, n_nodes)
        node_count_target = (degree_target > 0).float().sum(dim=-1)  # Nombre de nœuds actifs (batch_size,)
        edge_count_target = data.A.sum(dim=(-2, -1)) / 2  # Nombre total d'arêtes (batch_size,)

        # 2. Degree Loss (Différence entre les sommes des degrés des graphes)
        degree_loss = F.mse_loss(adj.sum(dim=-1), degree_target)

        # 3. Node Count Loss (Nombre de nœuds actifs par graphe)
        node_count_pred = (adj.sum(dim=-1) > 0).float().sum(dim=-1)  # Prédiction du nombre de nœuds actifs
        node_count_loss = F.mse_loss(node_count_pred, node_count_target)

        # 4. Edge Count Loss (Nombre total d'arêtes dans chaque graphe)
        edge_count_pred = adj.sum(dim=(-2, -1)) / 2  # Prédiction du nombre total d'arêtes
        edge_count_loss = F.mse_loss(edge_count_pred, edge_count_target)

        # 5. Density Loss (Distance entre les densités globales des graphes)
        density_pred = adj.sum(dim=(-2, -1)) / (adj.size(-1) ** 2)  # Densité des prédictions
        density_true = data.A.sum(dim=(-2, -1)) / (adj.size(-1) ** 2)  # Densité des graphes réels
        density_loss = torch.abs(density_pred - density_true).mean()

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Combinaison des pertes
        total_loss = (alpha * recon_loss +
                    beta * degree_loss +
                    gamma * node_count_loss +
                    delta * edge_count_loss +
                    epsilon * density_loss)
        recon = F.l1_loss(adj, data.A, reduction='mean')
        return total_loss, recon, kld, 0, recon_loss