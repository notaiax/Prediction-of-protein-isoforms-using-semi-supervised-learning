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
import pandas as pd
import numpy as np
from scipy.linalg import svd

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
# IMPORT PACKAGES
###############################################################

class IsoDataset(torch.utils.data.Dataset):
    def __init__(self):
        f_gtex_isoform = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        self.dset_isoform = f_gtex_isoform['expressions']
        self.tissues = list(f_gtex_isoform['tissue'][:])
        
        load_in_mem = False
        if load_in_mem:
            self.dset_isoform = np.array(self.dset_isoform)

    def __len__(self):
        return self.dset_isoform.shape[0]
        

    def __getitem__(self, idx):
        return self.dset_isoform[idx]

###############################################################
# EXTRACT IDXS TRAINING / TESTING
###############################################################
    
def extract_ids(full_data, subsample_size):

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
    #tissues_labels_sub = [tissues_labels_dict[idx] for idx in idxs_sub]

    return idxs_sub

###############################################################
# DATASETS
###############################################################

# Print progress
print("Loading data...")

# Load the entire and subset dataset
full_data = IsoDataset()

subsample_size = 0.15

idx_sub = extract_ids(full_data, subsample_size)

sub_dataset = Subset(full_data, indices = idx_sub)

#final_data = sub_dataset[:][:]
#print(final_data.shape)

sub_dataloader = DataLoader(sub_dataset, batch_size=len(sub_dataset), shuffle = False)

sub_dataset_full = []

for inputs in sub_dataloader:
    final_data = inputs

print(final_data.shape)

# convert to numpy

X = final_data.numpy()

N = X.shape[0]

Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))

U,S,Vh = svd(X,full_matrices=False)
V = Vh.T    

Z = Y @ V
Z = Z[:,:10000]
print(f"Z.shape: {Z.shape}")

Z_df = pd.DataFrame(Z)
Z_df.to_csv('Z_iso.tsv.gz', sep='\t', index=False, compression="gzip")

V_df = pd.DataFrame(V)
V_df.to_csv('V_iso.tsv.gz', sep='\t', index=False, compression="gzip")

rho = (S*S) / (S*S).sum()
print(rho[:2000])

rho_df = pd.DataFrame(rho)
rho_df.to_csv('rho_iso.tsv.gz', sep='\t', index=False, compression="gzip")



