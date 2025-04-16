# first gan trial from the github page https://github.com/lyeoni/pytorch-mnist-GAN/tree/master

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import preprocess_dataset
from torch.utils.data import DataLoader


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bs = 100  #  Batch Size


image_size = 50  # Image 50x50

train_data = preprocess_dataset('train', 100, 10, feature_extractor="encode_prompt")
print(train_data[0]['stats'].size())
print(train_data[0]['A'].size())
unique_values = torch.unique(train_data[0]['A'])
print(unique_values)

data_pairs = []
for data in train_data:
    embedding = torch.tensor(data['stats'], dtype=torch.float32)
    matrix = torch.tensor(data['A'], dtype=torch.float32).view(-1)  # Flatten the matrix 
    data_pairs.append((embedding, matrix))

train_loader = DataLoader(data_pairs, batch_size=bs, shuffle=True)  
    
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)


        return (torch.sigmoid(self.fc4(x)))  # Generated Image (normalized in [0, 1])
        

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# Network configuration
embedding_dim = embedding.size(1)  # Embedding size
z_dim = 700 # Size of the input for the generator
image_dim = image_size * image_size  # Dimensions of outputs 

# Initializations
G = Generator(g_input_dim = z_dim + embedding_dim, g_output_dim=image_dim).to(device)
D = Discriminator(image_dim).to(device)

criterion = nn.BCELoss()  
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

def D_train(embedding, real_matrix):
    D.zero_grad()

    # Real Input
    x_real = real_matrix  
    y_real = torch.ones(bs, 1).to(device)
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    #Generated Input
    z = torch.randn(bs, z_dim).to(device)
    squeezed_embedding = embedding.squeeze(1)
    z = torch.cat((z, squeezed_embedding), 1)
    x_fake = G(z)
    y_fake = torch.zeros(bs, 1).to(device)
    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # Loss computation
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

# Train the generator
def G_train(embedding):
    G.zero_grad()

    z = torch.randn(bs, z_dim).to(device)
    squeezed_embedding = embedding.squeeze(1)
    z = torch.cat((z, squeezed_embedding), 1)
    y = torch.ones(bs, 1).to(device)  # We want the generator to fool the discriminator

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

# Training
n_epoch = 10
for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    for batch_idx, (embedding, real_matrix) in enumerate(train_loader):
        embedding = embedding.to(device)
        real_matrix = real_matrix.to(device)

        # Train the Discriminator
        D_losses.append(D_train(embedding, real_matrix))

        # Train the Generator
        G_losses.append(G_train(embedding))


    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        epoch, n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

