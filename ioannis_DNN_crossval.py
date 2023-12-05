###############################################################
# IMPORT PACKAGES
###############################################################

print("Running script")

import torch
import numpy as np
import pandas as pd
import math
import torch
from torch import nn, Tensor

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.distributions.bernoulli import Bernoulli
from torch import optim
from torch.utils.data import Subset

import gzip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import h5py
import re
import collections
from collections import Counter
import matplotlib.pyplot as plt
import os

###############################################################
# IMPORT GENE AND ISOFORM FILES
###############################################################

class GtexDataset(torch.utils.data.Dataset):
    def __init__(self):
        f_gtex_gene = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_isoform = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        self.dset_gene = f_gtex_gene['expressions']
        self.dset_isoform = f_gtex_isoform['expressions']
        self.tissues = list(f_gtex_gene['tissue'][:])

        assert(self.dset_gene.shape[0] == self.dset_isoform.shape[0])
        
        load_in_mem = False
        if load_in_mem:
            self.dset_gene = np.array(self.dset_gene)
            self.dset_isoform = np.array(self.dset_isoform)

    def __len__(self):
        return self.dset_gene.shape[0]
        

    def __getitem__(self, idx):
        return self.dset_gene[idx], self.dset_isoform[idx]

###############################################################
# EXTRACT IDXS TRAINING / TESTING
###############################################################
    
def extract_ids(full_data, subsample_size, list_excluding_tissues, test_size):

    # Tissues
    tissues = full_data.tissues
    tissues = [byte_tissue.decode('utf-8') for byte_tissue in tissues]

    tissues_dict = {}
    i = 0
    for tissue in tissues:
        if tissue not in tissues_dict:
            tissues_dict[tissue] = i
            i += 1

    tissues_labels = [tissues_dict[tissue] for tissue in tissues]
    idxs = np.arange(len(tissues_labels))
    tissues_labels_dict = {idx:tissues_label for idx, tissues_label in zip(idxs, tissues_labels)}

    # Subsampling
    #idxs_sub = np.random.choice(idxs, size=int(len(tissues)*subsample_size), replace=False)
    _, idxs_sub = train_test_split(idxs, test_size=int(len(tissues)*subsample_size), stratify = tissues_labels, random_state=1)
    tissues_labels_sub = [tissues_labels_dict[idx] for idx in idxs_sub]
    
    # Excluding tissues
    tissues_labels_exclude = [tissues_dict[tissue] for tissue in list_excluding_tissues]

    idxs_exclude = []
    idxs_include = []
    tissues_labels_include = []

    # Iterate over labels and their corresponding indices
    for index, label in zip(idxs_sub, tissues_labels_sub):
        if label in tissues_labels_exclude:

            idxs_exclude.append(index)
            
        else:
            
            idxs_include.append(index)
            tissues_labels_include.append(label)

    train_idx, test_idx = train_test_split(idxs_include, test_size=test_size, stratify = tissues_labels_include, random_state=2)
    test_idx = test_idx + idxs_exclude

    train_labels = []
    for idx, label in zip(idxs_include,tissues_labels_include):
        if idx in train_idx:
            train_labels.append(label)

    return (train_idx, test_idx, train_labels)

###############################################################
# DATASETS
###############################################################

# Print progress
print("Loading data...")

# Load the entire and subset dataset
full_data = GtexDataset()
subsample_size = 0.3
list_excluding_tissues = ["Ovary","Stomach",'Thyroid', 'Kidney - Medulla']
test_size = 0.15

idx_tuple = extract_ids(full_data, subsample_size, list_excluding_tissues, test_size)
train_idx = idx_tuple[0]
test_idx = idx_tuple[1]
train_labels = idx_tuple[2]

train_dataset = Subset(full_data, indices = train_idx)
test_dataset = Subset(full_data, indices = test_idx)

###############################################################
# DATALOADERS
###############################################################

train_labels_tensor = torch.tensor(train_labels)
class_counts = torch.bincount(train_labels_tensor)
weights = 1.0 / class_counts[train_labels]

# Create the WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=64, sampler = sampler)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Usually, shuffling is not needed for testing

print("done")
