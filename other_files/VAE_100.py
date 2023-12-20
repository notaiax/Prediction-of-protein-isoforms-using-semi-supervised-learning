###############################################################
# IMPORT PACKAGES
###############################################################

from typing import *
import matplotlib
import matplotlib.pyplot as plt
#from IPython.display import Image, display, clear_output
import numpy as np
import seaborn as sns
import pandas as pd
import h5py
sns.set_style("whitegrid")
import requests

import math
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, LogNormal


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
from torch.distributions.bernoulli import Bernoulli

#import IsoDatasets
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

import gzip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from pathlib import Path

###############################################################
# VAE
###############################################################

def load_data_chunk(filename, chunk_size=1000):
    """ Load a chunk of data from a gzipped TSV file. """
    return pd.read_csv(filename, sep='\t', compression='gzip', chunksize=chunk_size)

def separate_ids_and_data(data):
    ids = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    return ids, data

def create_data_loaders(data_dir, batch_size, train_percent=0.8, load_in_mem=False):
    full_dataset = IsoDatasets.Archs4GeneExpressionDataset(data_dir, load_in_mem)
    train_size = int(train_percent * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def plot_line(tensor, axs, line_width=1.0):
    # Check if the input is a PyTorch tensor
    if isinstance(tensor, torch.Tensor):
        # Check if the tensor is on a CUDA device and move it to CPU if necessary
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Convert the Torch tensor to a NumPy array
        numeric_array = tensor.detach().numpy()
    else:
        # If it's not a tensor, assume it's already a NumPy array
        numeric_array = tensor

    # Plotting code remains the same
    axs.plot(numeric_array, linewidth=line_width)
    axs.set_title('Gene Expression Profile')
    axs.set_xlabel('Gene Index')
    axs.set_ylabel('Expression Level')

# Kaiming initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def is_nan(tensor):
    """ Check if a tensor is NaN """
    return torch.isnan(tensor).any()


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        # z = mu + sigma * epsilon
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        # Log probability for Gaussian distribution
        # log p(z) = -1/2 * [log(2*pi) + 2*log(sigma) + (z - mu)^2/sigma^2]
        return -0.5 * (torch.log(2 * torch.tensor(math.pi)) + 2 * torch.log(self.sigma) +
                       torch.pow(z - self.mu, 2) / torch.pow(self.sigma, 2))
    
    def count_csv_rows(filename):
        # If the file is gzip-compressed, decompress it first
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rt', newline='') as csvfile:
                row_count = sum(1 for row in csvfile)
        else:
            # Specify the correct encoding (e.g., 'utf-8', 'latin-1', etc.)
            encoding = 'utf-8'  # Change to the appropriate encoding if needed
            with open(filename, 'r', newline='', encoding=encoding) as csvfile:
                row_count = sum(1 for row in csvfile)
        return row_count



class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)

        dropout_rate = 0.2  # Dropout rate

        # Inference Network (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=2*latent_features)
        )

        # Generative Model (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=512, out_features=self.observation_features)
        )
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))

    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)

        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_params = self.decoder(z)
        px_params = px_params.view(-1, *self.input_shape) # reshape the output
        return LogNormal(px_params, 1.0) # Assuming variance of 1


    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        # flatten the input
        x = x.view(x.size(0), -1)

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}


    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)

        # sample the prior
        z = pz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'z': z}

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta

    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x)

        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl # <- your code here
        beta_elbo = log_px - self.beta * kl # <- your code here

        # loss
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}

        return loss, diagnostics, outputs

###############################################################
# LOAD ORIGINAL DATASET
###############################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        
        f_gtex_gene = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_gene_expression_norm_transposed.hdf5', mode='r')
        self.row_header = list(f_gtex_gene['row_names'][:])
        self.dset_gene = f_gtex_gene['expressions']

    def __len__(self):
        return self.dset_gene.shape[0]
        
    def __getitem__(self, idx):
        return self.dset_gene[idx]
    
dataset = Dataset()
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

###############################################################
# CREATE NEW DATASET
###############################################################

latent_features = 100
epochs = 101

#Row header
row_header = dataset.row_header
row_header = [row.decode('utf-8') for row in row_header]
row_header = np.array(row_header)
row_header = row_header.reshape(-1, 1)

#Col header
col_header = np.hstack((np.array(["id"]), np.arange(latent_features)))

# Load the model from a file
vae = VariationalAutoencoder(next(iter(dataloader))[1].shape, latent_features) # second parameter is number of latent features that the model whas trained on (backed up model is 100)
model_directory = "VAE/models/vae_LF_" + str(latent_features) + "_Epochs_" + str(epochs) + ".pth"
vae.load_state_dict(torch.load(model_directory, map_location=torch.device('cpu')))

Zs = []

# Example of making a training set that excludes samples from the brain and a test set with only samples from the brain
# If you have enough memory, you can load the dataset to memory using the argument load_in_mem=True

for x in tqdm(dataloader):
    x = x.to(device)

    # Forward pass through the VAE
    outputs = vae(x)
    z = outputs['z'].cpu().detach().numpy() # z is the latent space
    Zs.append(z)

# Concatenate the list of z values into a single array. Otherwise it separates by batches
Zs = np.concatenate(Zs, axis=0)

# The shape of Zs will be [total_num_batches * batch_size, latent_features]
print("Shape of Zs:", Zs.shape)

Zs = np.hstack((row_header, Zs))

print("Shape of Zs:", Zs.shape)

Zs_df = pd.DataFrame(Zs, columns = col_header)
Path("datasets_reduced").mkdir(exist_ok=True)
final_model_directory = "datasets_reduced/VAE_" + str(latent_features) + ".tsv.gz"
Zs_df.to_csv(final_model_directory, sep='\t', index=False, compression="gzip")