#!/usr/bin/env python
# coding: utf-8

# In[2]:


from typing import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
import seaborn as sns
import pandas as pd
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

import IsoDatasets
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from collections import defaultdict




import gzip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ## Helper functions

# In[3]:
# Define your color dictionary

color_dict = {
    'red': '#ab0000',   # Dark red
    'blue': '#0047ab'   # Dark blue
}

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

def plot_line(tensor, axs, title, xlabel, ylabel, line_width=1.0,):
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
    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)

def plot_and_save_losses(losses_dict, file_name):
    """
    Plots the training and validation losses on the same plot.

    Parameters:
    losses_dict (dict): A dictionary where keys are loss types (e.g., 'Training', 'Validation') 
                        and values are lists of loss values.
    """
    plt.figure(figsize=(15, 10))
    
    # Use darker blue and red colors
    colors = {
        'Training': color_dict.get('blue'),  # Dark blue
        'Validation': color_dict.get('red')  # Dark red
    }
    
    for type, losses in losses_dict.items():
        plt.plot(losses, label=f'{type} Loss', color=colors.get(type, 'black'), linewidth=.7, alpha=0.9)
    
    plt.title('Loss per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f'{file_name}.png')
    plt.savefig(f'{file_name}.svg', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()


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


# ## Load Data

# In[4]:


# hdf5 paths:
archs4_path = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/archs4_gene_expression_norm_transposed.hdf5"
gtex_gene_path = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/gtex_gene_expression_norm_transposed.hdf5"
gtex_isoform_path = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/gtex_isoform_expression_norm_transposed.hdf5"

# Here is an example of loading the Archs4 gene expression dataset and looping over it
# If you have about 12GB of memory, you can load the dataset to memory using the argument load_in_mem=True

archs4_train_dataloader, archs4_test_dataloader = create_data_loaders(
    "/dtu-compute/datasets/iso_02456/hdf5/",
    batch_size=64,
    train_percent=0.8,
    load_in_mem=False  # Set to False to avoid MemoryError
)

genes = next(iter(archs4_train_dataloader))
print(genes.shape)

train_loader = archs4_train_dataloader
test_loader = archs4_test_dataloader


# ## Building the model
# When defining the model the latent layer must act as a bottleneck of information, so that we ensure that we find a strong internal representation. We initialize the VAE with 1 hidden layer in the encoder and decoder using relu units as non-linearity.

# In[5]:


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


        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=128, out_features=2*latent_features) # <- note the 2*latent_features
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.observation_features)
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


# ## Training and Evaluation
# 
# ### Initialize the model, evaluator and optimizer

# ### Training Loop

# In[6]:


print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")


# Training loop

# In[7]:


# define the models, evaluator and optimizer
num_epochs = 100
latent_features = 2000

# Model Name
model_name = f"LF_{latent_features}_Epochs_{num_epochs}"
print(f"Model: {model_name}")

train_losses = []
val_losses = []

# VAE
vae = VariationalAutoencoder(genes[0].shape, latent_features)

# Evaluator: Variational Inference
beta = 1
vi = VariationalInference(beta=beta)
# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

vae = vae.to(device)

epoch = 0
pseudocount = 1e-8
while epoch < num_epochs:
    epoch += 1
    training_epoch_data = defaultdict(list)
    vae.train()

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
    for x in train_loader_tqdm:
        x = x.to(device)

        # Avoid LogNormal getting 0 values
        pseudocount = 1e-8
        x = x+pseudocount

        loss, diagnostics, outputs = vi(vae, x)
        train_losses.append(loss.item())

        # Check if loss is NaN
        if is_nan(loss):
            print("Error: Loss became NaN during training")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]

        train_loader_tqdm.set_postfix(loss=loss.item())

    if is_nan(loss):
        break  # Stop training if loss is NaN

    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    with torch.no_grad():
        vae.eval()
        validation_epoch_data = defaultdict(list)

        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch}/{num_epochs} [Test]")
        for x in test_loader_tqdm:
            x = x.to(device)
            x = x+pseudocount

            loss, diagnostics, outputs = vi(vae, x)
            val_losses.append(loss.item())

            # Check if loss is NaN
            if is_nan(loss):
                print("Error: Loss became NaN during validation")
                break

            for k, v in diagnostics.items():
                validation_epoch_data[k] += [v.mean().item()]

            test_loader_tqdm.set_postfix(loss=loss.item())

        if is_nan(loss):
            break  # Stop validation if loss is NaN

        for k, v in validation_epoch_data.items():
            validation_data[k] += [np.mean(validation_epoch_data[k])]


# Losses Plot

# In[ ]:


losses_dict = {
    "Training": train_losses,
    "Validation": val_losses
}
plot_and_save_losses(losses_dict, f"plots/{model_name}_LossesTrainAndVal")

# Save and Load model

# In[20]:


# Save the model to a file
torch.save(vae.state_dict(), f'models/vae_{model_name}.pth')


# ## Evaluate model

# - KL Divergence
#     - For VAEs, it's used to quantify how much the learned distribution (the posterior) deviates from the prior distribution, so it acts a  regularizer, ensuring that the variational distribution doesn't stray too far from the prior.
#     - Interpretation: A KL divergence of 0 indicates that the two distributions are identical. The greater the KL divergence, the more the distributions differ.
#     - Usage in VAEs: In VAEs, the KL divergence is used as a regularizer in the loss function to ensure that the distribution of the latent variables (encoded representations) doesn't deviate too much from a prior distribution (often a Gaussian). This helps in generalizing the model and avoiding overfitting to the training data.
# - ELBO:
#     - It's essentially a measure of how well the decoder can reconstruct the input from the latent variables.
#     - How do we know if ELBO is good?
#         - Summary: Look for improvement over epochs, combine with Qualitative Analysis on reconstructed data, and compare with baseline.
#         - Depends on the Data: The ELBO is influenced by the complexity and characteristics of your dataset. Different datasets will naturally lead to different ranges of ELBO values. High-dimensional, complex data might result in lower ELBOs compared to simpler, lower-dimensional data.
#         - Relative, not Absolute: The ELBO is more useful as a relative measure than an absolute one. This means you generally use ELBO to compare different models or configurations on the same dataset. An improvement in ELBO from one model iteration to the next can indicate progress.
#         - Negative Values: ELBO values are often negative (since they involve log probabilities), and a higher (less negative) value is typically better. But without comparing to a baseline or other models, it's hard to label a specific number as good.
#         - Model and Task Specific: The interpretation of ELBO can also depend on the specific use case of the model. For instance, if you're using a VAE for generative purposes, the quality of generated samples might be more important than the ELBO value itself.
#         - Balance Between Components: ELBO combines reconstruction loss (how well the model can recreate input data) and the KL divergence (how much the model's latent variable distribution deviates from a prior distribution). The balance between these two can vary. A "good" ELBO in one context might mean excellent reconstruction at the cost of higher KL divergence, or vice versa.
#         - Baseline Comparison: Compare your model's ELBO to that of a baseline model or a simpler version of your current model.
#     - Improving the ELBO score means the model is getting better at compressing the data into a meaningful, compact form while also keeping its representations realistic and generalizable.
#     - KL divergence it's part of the ELBO Loss together with the Reconstruction Loss that is how well the model can recreate the input data from its internal representation.

# In[11]:


# Set up a figure and axis for the plots
plt.figure(figsize=(12, 6))

# Plot ELBO
plt.subplot(1, 2, 1)
plt.plot(training_data['elbo'], label='Training ELBO', color=color_dict.get('blue'))
plt.plot(validation_data['elbo'], label='Validation ELBO', color=color_dict.get('red'))
plt.xlabel('Epochs')
plt.ylabel('ELBO')
plt.title('ELBO over Epochs')
plt.legend()

# Plot KL Divergence
plt.subplot(1, 2, 2)
plt.plot(training_data['kl'], label='Training KL Divergence', color=color_dict.get('blue'))
plt.plot(validation_data['kl'], label='Validation KL Divergence', color=color_dict.get('red'))
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.title('KL Divergence over Epochs')
plt.legend()

# Save the figure instead of displaying it
plt.savefig(f'plots/{model_name}_ELBO_and_KL_Divergence.png') 
plt.savefig(f'plots/{model_name}_ELBO_and_KL_Divergence.svg', bbox_inches='tight', transparent="True", pad_inches=0)


# In[23]:


# Sample from the prior
sampled_data = vae.sample_from_prior(batch_size=10)
generated_samples = sampled_data['px'].mean  # Assuming Bernoulli distribution

generated_samples.shape


# In[24]:


# Sample a batch from your data
x = next(iter(test_loader))
x = x.to(device)

# Forward pass through the VAE
outputs = vae(x)
z = outputs['z'].cpu().detach().numpy() # z is the latent space
print(f"Latent space size: {z.shape[1]}")
print(z.shape)


# In[25]:


# Forward pass through the VAE
reconstructed_data = outputs['px'].mean  # Assuming Bernoulli distribution

# Compare original 'x' and 'reconstructed_data' visually or with a metric
reconstructed_data.shape


# ### Plot VAE data

# Generated Gene VS Real Gene

# In[29]:


# Create a figure with multiple subplots
fig, axs = plt.subplots(4, 1, figsize=(15, 20))  # 4 rows, 1 column

# Plot each line on its respective subplot
plot_line(genes[0], axs[0], title='Original Gene Expression Profile Sample 1', xlabel='Gene Index', ylabel='Expression Level', line_width=0.2)
plot_line(reconstructed_data[0], axs[1], title='Generated Gene Expression Profile Sample 1', xlabel='Gene Index', ylabel='Expression Level', line_width=0.2)
plot_line(genes[1], axs[2], title='Original Gene Expression Profile Sample 2', xlabel='Gene Index', ylabel='Expression Level', line_width=0.2)
plot_line(reconstructed_data[1], axs[3], title='Generated Gene Expression Profile Sample 2', xlabel='Gene Index', ylabel='Expression Level', line_width=0.2)


# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig(f'plots/{model_name}_GeneratedVsRealGene.png') 
plt.savefig(f'plots/{model_name}_GeneratedVsRealGene.svg', bbox_inches='tight', transparent="True", pad_inches=0)



# We can see that our model generates genes with values that are wither 0 or 1, while a real gene have values that go from 0 to 14. Further steps include checking that model architecture is correct, and improve model by augmenting latent space and epochs.

# Latent space sample

# In[31]:


# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(15, 10))  # 2 rows, 1 column

# Plot each line on its respective subplot
plot_line(z[0], axs[0], title='Latent Space Sample 1', xlabel='Latent features', ylabel='Expression Level', line_width=1)
plot_line(z[1], axs[1], title='Latent Space Sample 2', xlabel='Latent features', ylabel='Expression Level', line_width=1)


# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig(f'plots/{model_name}_LatentSpaceVis.png') 
plt.savefig(f'plots/{model_name}_LatentSpaceVis.svg', bbox_inches='tight', transparent="True", pad_inches=0)


print("EXECUTION FINISHED WITHOUT PROBLEMS")

