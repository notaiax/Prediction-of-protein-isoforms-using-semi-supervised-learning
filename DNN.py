###############################################################
# IMPORT PACKAGES
###############################################################

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

from sklearn.model_selection import train_test_split, StratifiedKFold
import h5py
from collections import Counter
import matplotlib.pyplot as plt
import os

###############################################################
# EXTRACT IDXS TRAINING / TESTING
###############################################################

# Function
 
def extract_ids(tissues, list_excluding_tissues, test_size, num_folds):

    # Tissues
    tissues_dict = {}
    i = 0
    for tissue in tissues:
        if tissue not in tissues_dict:
            tissues_dict[tissue] = i
            i += 1
    tissues_labels = [tissues_dict[tissue] for tissue in tissues]
    idxs = np.arange(len(tissues_labels))
    tissues_labels_dict = {idx:tissues_label for idx, tissues_label in zip(idxs, tissues_labels)}

    # Full sample
    idx = idxs
    tissue = tissues_labels

    # Excluding tissues
    tissue_exclude = [tissues_dict[tissue] for tissue in list_excluding_tissues]

    idx_include = []
    idx_exclude = []

    for idx, tissue in zip(idx, tissue):
        if tissue in tissue_exclude:
            idx_exclude.append(idx)
        else:
            idx_include.append(idx)

    tissue_include = [tissues_labels_dict[idx] for idx in idx_include]

    # Train&valid / Test
    idx_train_valid, idx_test = train_test_split(idx_include, test_size=int(len(idx_include)*test_size), stratify = tissue_include, random_state=1)
    idx_test = idx_test + idx_exclude

    tissue_train_valid = [tissues_labels_dict[idx] for idx in idx_train_valid]

    # Train / Valid
    stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    idx_train_list = []
    idx_valid_list = []

    for idx_train, idx_valid in stratified_kfold.split(idx_train_valid, tissue_train_valid):
        
        idx_train_list.append(idx_train)
        idx_valid_list.append(idx_valid)
    
    #Return
    return (idx_test, idx_train_list, idx_valid_list, tissues_labels_dict)

# Load tissues
tissues_pd = pd.read_csv("tissues.tsv.gz", sep = "\t", compression = "gzip", header = None)
tissues = tissues_pd.values
tissues = np.squeeze(tissues, axis=1)
tissues = tissues.tolist()[1:]

list_excluding_tissues = ["Brain - Amygdala", "Brain - Anterior cingulate cortex (BA24)",
                          "Brain - Caudate (basal ganglia)", "Brain - Cerebellar Hemisphere", "Brain - Cerebellum",
                          "Brain - Cortex", "Brain - Frontal Cortex (BA9)", "Brain - Hippocampus", "Brain - Hypothalamus",
                          "Brain - Nucleus accumbens (basal ganglia)", "Kidney - Medulla"]
test_size = 0.18
num_folds = 5

idx_test, idx_train_list, idx_valid_list, tissues_labels_dict = extract_ids(tissues, list_excluding_tissues, test_size, num_folds)

###############################################################
# CREATE CLASS DATASET
###############################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model, num_comp_pca, num_latent_features):
        f_gtex_isoform = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_isoform_expression_norm_transposed.hdf5', mode='r')
        f_gtex_gene = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_pca = h5py.File('pca_data.hdf5', mode='r')
        #f_gtex_VAE = h5py.File("VAE_data.hdf5", mode="r")

        self.dset_isoform = f_gtex_isoform['expressions']
        self.dset_gene = f_gtex_gene['expressions']
        self.dset_pca = f_gtex_pca['expressions'][:,:num_comp_pca]
        #self.dset_VAE = f_gtex_VAE['expressions'][:,:num_latent_features]

        assert(self.dset_gene.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_pca.shape[0] == self.dset_isoform.shape[0])
        #assert(self.dset_VAE.shape[0] == self.dset_isoform.shape[0])
        
        if model == "gene":
            self.x = self.dset_gene
        elif model == "pca":
            self.x = self.dset_pca
        elif model == "VAE":
            self.x = self.dset_VAE

    def __len__(self):
        return self.dset_isoform.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.dset_isoform[idx]

###############################################################
# LOAD DATA
###############################################################

#models = ["gene","pca","VAE"]
models = ["gene","pca"]
num_comp_pca = 100
num_latent_features = 100

train_loaders = []
valid_loaders = []
test_loaders = []

for model in models:
    
    full_data = Dataset(model, num_latent_features, num_comp_pca)

    ###############################################################
    # DATASETS
    ###############################################################

    train_dataset = Subset(full_data, indices = idx_train_list[0])
    valid_dataset = Subset(full_data, indices = idx_valid_list[0])
    test_dataset = Subset(full_data, indices = idx_test)

    ###############################################################
    # DATALOADERS
    ###############################################################

    # Balance train_loader
    tissue_train = [tissues_labels_dict[idx] for idx in idx_train_list[0]]
    tissue_train_tensor = torch.tensor(tissue_train)
    class_counts = torch.bincount(tissue_train_tensor)
    weights = 1.0 / class_counts[tissue_train]
    train_sampler = WeightedRandomSampler(weights, len(weights))

    # Dataloaders
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Appending

    train_loaders.append(train_loader)
    valid_loaders.append(valid_loader)
    test_loaders.append(test_loader)

###############################################################
# TRAINING / VALIDATION
###############################################################

NUM_FEATURES = next(iter(train_loaders[0]))[0].shape[1]
NUM_OUTPUT = next(iter(train_loaders[0]))[1].shape[1]

# define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        activation_fn = nn.ReLU
        num_hidden = 40
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, num_hidden),
            activation_fn(),
            nn.Linear(num_hidden, NUM_OUTPUT)
        )

    def forward(self, x):
        return self.net(x)

model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-1)\


### BASELINE MODEL
baseline = pd.read_csv("baseline.csv")
avgs = torch.tensor(baseline["averages"].to_numpy()).to(device)

### TRAINING / TESTING

num_epochs = 100
model.train()
train_losses = []
test_losses = []

train_losses_baseline = []
test_losses_baseline = []

for epoch in range(num_epochs):
    print(f"epoch number {epoch}")

    total_loss = 0
    total_loss_baseline = 0
    num_batches = 0

    for inputs, targets in train_loaders[0]:
        print(f"\tbatch number {num_batches}")
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass, compute gradients, perform one training step.
        # Your code here!

        # Forward pass.
        output = model(inputs)

        # Compute loss.
        loss = loss_fn(output, targets)
        print(f"\tloss: {loss}")

        # Clean up gradients from the model.
        optimizer.zero_grad()

        # Compute gradients based on the loss from the current batch (backpropagation).
        loss.backward()

        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()

        # Increment step counter
        total_loss += loss
        total_loss_baseline += loss_fn(avgs.repeat(targets.shape[0], 1), targets)
        num_batches += 1

    train_loss = total_loss / num_batches
    train_loss_baseline = total_loss_baseline / num_batches

    print(f"\ttraining loss: {train_loss}")
    train_losses.append(train_loss)
    train_losses_baseline.append(train_loss_baseline)

    with torch.no_grad():
        model.eval()
        total_loss = 0
        total_loss_baseline = 0
        num_batches = 0

        for inputs, targets in test_loaders[0]:
            print(f"\ttesting batch number {num_batches}")
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            print(f"\tloss: {loss}")
            num_batches += 1
            
            total_loss += loss
            total_loss_baseline += loss_fn(avgs.repeat(targets.shape[0], 1), targets)

        test_loss = total_loss / num_batches
        test_loss_baseline = total_loss_baseline / num_batches

        print(f"\ttesting loss: {test_loss}")
        test_losses.append(test_loss)
        test_losses_baseline.append(test_loss_baseline)

        model.train()
        
print("Finished training.")

def new_file_name(name: str, extension: str):
    num = 1
    files = {f for f in os.listdir('.') if os.path.isfile(f)}
    while f"{name}_{num}.{extension}" in files:
        num += 1
    
    return f"{name}_{num}.{extension}"

plt.figure()
plt.plot([e.cpu().detach().numpy() for e in train_losses][2:], label="Train loss")
plt.plot([e.cpu().detach().numpy() for e in test_losses][2:], label="Test loss")
plt.plot([e.cpu().detach().numpy() for e in train_losses_baseline][2:], label="Baseline train loss")
plt.plot([e.cpu().detach().numpy() for e in test_losses_baseline][2:], label="Baseline test loss")
plt.legend()
plt.title("Losses")
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.show()
fname = new_file_name("loss", "png")
plt.savefig(fname, dpi=400)
print(f"Saved loss plot to {fname}")

print(f"Train losses: {train_losses}")
print(f"Baseline train losses: {train_losses_baseline}")
print(f"Test losses: {test_losses}")
print(f"Baseline test losses: {test_losses_baseline}")