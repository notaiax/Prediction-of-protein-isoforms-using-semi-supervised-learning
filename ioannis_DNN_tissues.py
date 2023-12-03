###############################################################
# IMPORT PACKAGES
###############################################################

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

###############################################################
# PCA LOADING
###############################################################
###############################################################
# PCA LOADING
###############################################################

class PCADataset(torch.utils.data.Dataset):
    def __init__(self, num_comp_pca):
        f_gtex_pca = h5py.File('pca.hdf5', mode='r')
        f_gtex_isoform = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        self.dset_isoform = f_gtex_isoform['expressions']
        self.tissues = list(f_gtex_isoform['tissue'][:])
        self.dset_pca = f_gtex_pca['expressions']
        self.dset_pca = self.dset_pca[:,:num_comp_pca]

        assert(self.dset_pca.shape[0] == self.dset_isoform.shape[0])
        
        load_in_mem = False
        if load_in_mem:
            self.dset_pca = np.array(self.dset_pca)
            self.dset_isoform = np.array(self.dset_isoform)

    def __len__(self):
        return self.dset_pca.shape[0]
        

    def __getitem__(self, idx):
        return self.dset_pca[idx], self.dset_isoform[idx]

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
num_comp_pca = 100
PCA_full_data = PCADataset(num_comp_pca)
subsample_size = 0.3
list_excluding_tissues = ["Ovary","Stomach",'Thyroid', 'Kidney - Medulla']
test_size = 0.15

idx_tuple = extract_ids(PCA_full_data, subsample_size, list_excluding_tissues, test_size)
train_idx = idx_tuple[0]
test_idx = idx_tuple[1]
train_labels = idx_tuple[2]

PCA_train_dataset = Subset(PCA_full_data, indices = train_idx)
PCA_test_dataset = Subset(PCA_full_data, indices = test_idx)

###############################################################
# DATALOADERS
###############################################################

train_labels_tensor = torch.tensor(train_labels)
class_counts = torch.bincount(train_labels_tensor)
weights = 1.0 / class_counts[train_labels]

# Create the WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights))

PCA_train_loader = DataLoader(PCA_train_dataset, batch_size=64, sampler = sampler)
PCA_test_loader = DataLoader(PCA_test_dataset, batch_size=64, shuffle=False)  # Usually, shuffling is not needed for testing

print("done")

###############################################################
# TRAINING / TESTING
###############################################################

NUM_FEATURES = 2000
NUM_OUTPUT = 156958

# define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        activation_fn = nn.ReLU
        num_hidden = (NUM_FEATURES + NUM_OUTPUT) // 4
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, num_hidden),
            activation_fn(),
            nn.Linear(num_hidden, NUM_OUTPUT),
            activation_fn()
        )

    def forward(self, x):
        return self.net(x)

model = Net()
# device = torch.device('cuda')  # use cuda or cpu
# model.to(device)
print(model)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# # Test the forward pass with dummy data
# out = model(torch.randn(2, 3, 32, 32, device=device))
# print("Output shape:", out.size())
# print(f"Output logits:\n{out.detach().cpu().numpy()}")
# print(f"Output probabilities:\n{out.softmax(1).detach().cpu().numpy()}")




# batch_size = 64
# num_epochs = 10
# validation_every_steps = 500

# step = 0
# model.train()

# train_accuracies = []
# valid_accuracies = []

# for epoch in range(num_epochs):

#     train_accuracies_batches = []

#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)

#         # Forward pass, compute gradients, perform one training step.
#         # Your code here!

#         # Forward pass.
#         output = model(inputs)

#         # Compute loss.
#         loss = loss_fn(output, targets)

#         # Clean up gradients from the model.
#         optimizer.zero_grad()

#         # Compute gradients based on the loss from the current batch (backpropagation).
#         loss.backward()

#         # Take one optimizer step using the gradients computed in the previous step.
#         optimizer.step()

#         # Increment step counter
#         step += 1

#         # Compute accuracy.
#         predictions = output.max(1)[1]
#         train_accuracies_batches.append(accuracy(targets, predictions))

#         if step % validation_every_steps == 0:

#             # Append average training accuracy to list.
#             train_accuracies.append(np.mean(train_accuracies_batches))

#             train_accuracies_batches = []

#             # Compute accuracies on validation set.
#             valid_accuracies_batches = []
#             total_loss = 0

#             with torch.no_grad():
#                 model.eval()
#                 for inputs, targets in test_loader:
#                     inputs, targets = inputs.to(device), targets.to(device)
#                     output = model(inputs)
#                     loss = loss_fn(output, targets)
#                     total_loss += loss

#                     predictions = output.max(1)[1]

#                     # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
#                     valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))

#                 model.train()

#             # Append average validation accuracy to list.
#             valid_accuracies.append(np.sum(valid_accuracies_batches) / len(test_set))

#             print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
#             print(f"             test accuracy: {valid_accuracies[-1]}")
#             print(f"             total loss: {total_loss}")

# print("Finished training.")