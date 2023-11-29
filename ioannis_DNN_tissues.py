from typing import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
# import seaborn as sns
import pandas as pd
# sns.set_style("whitegrid")
import requests

import math
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
from torch.distributions.bernoulli import Bernoulli
from torch import optim

import gzip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import h5py
import re

gtex_tissue_list = "tissue_dict_list.pth"

num_rows_to_load = 100  # specify the number of rows you want to load

# Print progress
print("Loading data...")

# Load the entire and subset dataset
full_data = GtexDataset()
subset_data = PartialDataset(full_data, )
# Print progress
print("Data loaded.")

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")

tissue_labels = torch.stack(gtex_tissue_list)

train_idx, test_idx = train_test_split(np.arange(len(tissue_labels)), test_size=0.2, stratify = gtex_tissue_list, random_state=2)

train_dataset = Subset(subset_data, indices = train_idx)

test_dataset = Subset(subset_data, indices = test_idx)
##############################################################

labels_train = tissue_labels[train_idx]

labels_train = tissue_labels.argmax(dim = 1)

class_counts = torch.bincount(labels_train)

# Calculate the sample weights
weights = 1.0 / class_counts[labels_train]

# Create the WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights))

# Print progress
print("Data split.")

# # Create dataset instances for training and testing
# print("Creating training and testing datasets...")
# gene_train_dataset = GeneExpressionDataset(train_data.iloc[:, 1:], train_data.iloc[:, 0])
# gene_test_dataset = GeneExpressionDataset(test_data.iloc[:, 1:], test_data.iloc[:, 0])

# Print progress
print("Datasets created.")

gene_train_loader = DataLoader(train_dataset, batch_size=64, sampler = sampler)
gene_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Usually, shuffling is not needed for testing

# Print progress
print("Data loaders created.")