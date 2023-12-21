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
from dataclasses import dataclass
from typing import Callable, Any
from enum import Enum
import csv

###############################################################
# UTILITIES
###############################################################
def new_file_name(name: str, extension: str):
    num = 1
    files = {f for f in os.listdir('.') if os.path.isfile(f)}
    while f"{name}_{num}.{extension}" in files:
        num += 1
    
    return f"{name}_{num}.{extension}"

# Redefine print so that we get real-time logs
log_fname = new_file_name("log", "txt")

def print(msg):
    with open(log_fname, "a") as file:
        file.write(str(msg) + "\n")

print("Beginning log")

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

    # Full sample
    tissue = [tissues_dict[tissue] for tissue in tissues]
    idx = np.arange(len(tissue))

    # Train / Test outer level
    idx_train_out, idx_test_out = train_test_split(idx, test_size=int(len(idx)*test_size), stratify = tissue, random_state=1)
    tissue_train_out = [tissue[idx] for idx in idx_train_out]

    # Train / Test 2nd level
    stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    idx_train_in = []
    idx_test_in = []

    for idx_train, idx_test in stratified_kfold.split(idx_train_out, tissue_train_out):
        
        # Excluding
        tissue_exclude = [tissues_dict[tissue] for tissue in list_excluding_tissues]
        tissue_train = [tissue[idx] for idx in idx_train]

        idx_include = []
        idx_exclude = []

        for idx, tissue_label in zip(idx_train, tissue_train):
            if tissue_label in tissue_exclude:
                idx_exclude.append(idx)
            else:
                idx_include.append(idx)

        idx_train = idx_include

        idx_train_in.append(idx_train)
        idx_test_in.append(idx_test)

    #Return
    return (idx_train_out, idx_test_out, idx_train_in, idx_test_in, tissue)

# Load tissues
"""
tissues_pd = pd.read_csv("tissues.tsv.gz", sep = "\t", compression = "gzip", header = None)
tissues = tissues_pd.values
tissues = np.squeeze(tissues, axis=1)
tissues = tissues.tolist()[1:]
"""
f_gtex_gene = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_gene_expression_norm_transposed.hdf5', mode='r')
tissues = list(f_gtex_gene['tissue'][:])
tissues = [byte_tissue.decode('utf-8') for byte_tissue in tissues]

list_excluding_tissues = [
    "Brain - Amygdala",
    "Brain - Anterior cingulate cortex (BA24)",
    "Brain - Caudate (basal ganglia)",
    "Brain - Cerebellar Hemisphere",
    "Brain - Cerebellum",
    "Brain - Cortex",
    "Brain - Frontal Cortex (BA9)",
    "Brain - Hippocampus",
    "Brain - Hypothalamus",
    "Brain - Nucleus accumbens (basal ganglia)"
]
                               
test_size = 0.2
num_folds = 5

idx_train_out, idx_test_out, idx_train_in, idx_test_in, tissue = extract_ids(tissues, list_excluding_tissues, test_size, num_folds)

###############################################################
# CREATE CLASS DATASET
###############################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model, num_comp_pca):
        f_gtex_isoform = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_isoform_expression_norm_transposed.hdf5', mode='r')
        f_gtex_gene = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_pca = h5py.File('datasets_reduced/PCA.hdf5', mode='r')
        f_gtex_VAE_100 = h5py.File("datasets_reduced/VAE_100.hdf5", mode="r")
        f_gtex_VAE_500 = h5py.File("datasets_reduced/VAE_500.hdf5", mode="r")

        self.dset_isoform = f_gtex_isoform['expressions']
        self.dset_gene = f_gtex_gene['expressions']
        self.dset_pca = f_gtex_pca['expressions'][:,:num_comp_pca]
        self.dset_VAE_100 = f_gtex_VAE_100['expressions']
        self.dset_VAE_500 = f_gtex_VAE_500['expressions']

        assert(self.dset_gene.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_pca.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_VAE_100.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_VAE_500.shape[0] == self.dset_isoform.shape[0])
        
        match model:
            case "original":
                self.x = self.dset_gene
            case "pca":
                self.x = self.dset_pca
            case "VAE_100":
                self.x = self.dset_VAE_100
            case "VAE_500":
                self.x = self.dset_VAE_500

    def __len__(self):
        return self.dset_isoform.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.dset_isoform[idx]

###############################################################
# LOAD DATA
###############################################################

models = ["original","pca","VAE_100","VAE_500"]
num_comp_pca = 100
batch_strat = True

train_out = []
test_out = []
train_in = []
test_in = []

for model in models:

    ## INNER

    dataloader_train_in_list = []
    dataloader_test_in_list = []

    for fold in range(num_folds):
    
        full_data = Dataset(model, num_comp_pca)

        ###############################################################
        # DATASETS
        ###############################################################

        dataset_train_in = Subset(full_data, indices = idx_train_in[fold])
        dataset_test_in = Subset(full_data, indices = idx_test_in[fold])

        ###############################################################
        # DATALOADERS
        ###############################################################

        batch_size = 64
        if batch_strat:

            # Balance train_loader
            tissue_train = [tissue[idx] for idx in idx_train_in[fold]]
            tissue_train_tensor = torch.tensor(tissue_train)
            class_counts = torch.bincount(tissue_train_tensor)
            weights = 1.0 / class_counts[tissue_train]
            train_sampler = WeightedRandomSampler(weights, len(weights))

            # Dataloaders
            

            dataloader_train_in = DataLoader(dataset_train_in, batch_size=batch_size, sampler = train_sampler)
            dataloader_test_in = DataLoader(dataset_test_in, batch_size=batch_size, shuffle=False)
        
        else:

            dataloader_train_in = DataLoader(dataset_train_in, batch_size=batch_size, shuffle=True)
            dataloader_test_in = DataLoader(dataset_test_in, batch_size=batch_size, shuffle=False)

        # Appending
        dataloader_train_in_list.append(dataloader_train_in)
        dataloader_test_in_list.append(dataloader_test_in)
    
    ## OUTER

    ###############################################################
    # DATASETS
    ###############################################################

    dataset_train_out = Subset(full_data, indices = idx_train_out)
    dataset_test_out = Subset(full_data, indices = idx_test_out)

    ###############################################################
    # DATALOADERS
    ###############################################################

    if batch_strat:

        # Balance train_loader
        tissue_train = [tissue[idx] for idx in idx_train_out]
        tissue_train_tensor = torch.tensor(tissue_train)
        class_counts = torch.bincount(tissue_train_tensor)
        weights = 1.0 / class_counts[tissue_train]
        train_sampler = WeightedRandomSampler(weights, len(weights))

        # Dataloaders
        batch_size = 64

        dataloader_train_out = DataLoader(dataset_train_out, batch_size=batch_size, sampler = train_sampler)
        dataloader_test_out = DataLoader(dataset_test_out, batch_size=batch_size, shuffle=False)
    
    else:
        
        dataloader_train_out = DataLoader(dataset_train_out, batch_size=batch_size, shuffle = True)
        dataloader_test_out = DataLoader(dataset_test_out, batch_size=batch_size, shuffle=False)


    ## APPENDING
    train_in.append(dataloader_train_in_list)
    test_in.append(dataloader_test_in_list)
    train_out.append(dataloader_train_out)
    test_out.append(dataloader_test_out)

###############################################################
# 6-FOLD CROSS-VALIDATION
###############################################################
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_OUTPUT = next(iter(train_in[0][0]))[1].shape[1]
print(f"NUM_OUTPUT: {NUM_OUTPUT}")
NUM_FOLDS = 5
NUM_EPOCHS = 100

# Network model
class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()

        activation_fn = nn.ReLU
        num_hidden = 20
        self.net = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            activation_fn(),
            nn.Linear(num_hidden, NUM_OUTPUT)
        )

    def forward(self, x):
        return self.net(x)

# Baseline model
class BaselineModel(Callable):
    BASELINE_CSV = pd.read_csv("baseline.csv")
    AVGS = torch.tensor(BASELINE_CSV["averages"].to_numpy()).to(DEVICE)

    def __call__(self, inputs):
        return self.AVGS.repeat(inputs.shape[0], 1)
    
    def to(self, _):
        pass

    def train(self):
        pass

    def eval(self):
        pass

@dataclass
class ModelInfo:
    model_type: str
    create_model: Callable[[], Callable]
    requested_loader: int
    is_baseline: bool = False

model_infos = [
    ModelInfo("Original", lambda: Net(18965), 0),
    ModelInfo("PCA", lambda: Net(100), 1),
    ModelInfo("VAE 100", lambda: Net(100), 2),
    ModelInfo("VAE 500", lambda: Net(500), 3),
    ModelInfo("Baseline", lambda: BaselineModel(), 0, is_baseline = True)
]


@dataclass
class ModelResult:
    train_losses: list[Any]
    test_loss: Any

@dataclass
class OuterModelResult:
    min_train_losses: list[Any]
    average_test_loss: Any

average_model_results = []
print("Beginning training")
for model_idx, model_info in enumerate(model_infos):
    # For each model, run 5 folds, and find the average test error

    # Extract model info
    model_type = model_info.model_type
    print(f"Using model {model_type}")
    requested_loader = model_info.requested_loader

    fold_model_results: list[ModelResult] = []
    for fold in range(NUM_FOLDS):
        print(f"Fold number {fold}")
        model = model_info.create_model()

        model.to(DEVICE)
        # print(model)

        loss_fn = nn.MSELoss()

        if not model_info.is_baseline:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

        ### TRAINING
        model.train()

        i = 0
        train_losses = []
        for epoch in range(NUM_EPOCHS):
            # print(f"epoch number {epoch}")

            total_epoch_loss = 0
            batch_num = 0

            for inputs, targets in train_in[requested_loader][fold]:
                # print(f"\tbatch number {batch_num}")
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # Forward pass, compute gradients, perform one training step.
                # Your code here!

                # Forward pass.
                output = model(inputs)

                # Compute loss.
                loss = loss_fn(output, targets)
                # print(f"\tloss: {loss}")

                if not model_info.is_baseline:
                    # Clean up gradients from the model.
                    optimizer.zero_grad()

                    # Compute gradients based on the loss from the current batch (backpropagation).
                    loss.backward()

                    # Take one optimizer step using the gradients computed in the previous step.
                    optimizer.step()

                # Increment step counter
                total_epoch_loss += loss
                batch_num += 1

            train_epoch_loss = total_epoch_loss / batch_num

            # print(f"\ttraining loss: {train_loss}")
            train_losses.append(train_epoch_loss)
            
        if not model_info.is_baseline:
            torch.save(model.state_dict(), "models_100epochssimple/" + str(model_type) + "_" + str(fold) + ".pth")
        
        print("Finished training.")

        ### TESTING
        with torch.no_grad():
            model.eval()
            total_loss = 0
            batch_num = 0

            for inputs, targets in test_in[requested_loader][fold]:
                # print(f"\ttesting batch number {batch_num}")
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                output = model(inputs)
                loss = loss_fn(output, targets)
                # print(f"\tloss: {loss}")
                batch_num += 1
                
                total_loss += loss
            
            test_loss = total_loss / batch_num
            
        
        fold_model_results.append(ModelResult(train_losses, test_loss))
    
    average_test_loss = sum(e.test_loss for e in fold_model_results) / NUM_FOLDS
    min_train_losses = min(fold_model_results, key = lambda e: e.test_loss).train_losses
    
    average_model_results.append((model_info, OuterModelResult(min_train_losses, average_test_loss)))


# Save results
train_results_dict = {
    info.model_type: [e.cpu().detach().numpy() for e in result.min_train_losses] for (info, result) in average_model_results
}
inner_train_loss_file = new_file_name("inner_train_losses", "csv")
pd.DataFrame(train_results_dict).to_csv(inner_train_loss_file, index=False)
print(f"Wrote inner train loss to {inner_train_loss_file}")

test_results_dict = {
    info.model_type: [result.average_test_loss.cpu().detach().numpy()] for (info, result) in average_model_results
}
inner_test_loss_file = new_file_name("inner_test_losses", "csv")
pd.DataFrame(test_results_dict).to_csv(inner_test_loss_file, index=False)
print(f"Wrote inner test loss to {inner_test_loss_file}")

###############################################################
# OUTER TESTING
###############################################################
### SELECTION
best_model_info, best_model_results = min(average_model_results, key = lambda e: e[1].average_test_loss)
print(f"Model {best_model_info.model_type} had the best performance with an average loss of {best_model_results.average_test_loss}")

best_model = best_model_info.create_model()
best_model = best_model.to(DEVICE)
best_model_loader = best_model_info.requested_loader
best_model_type = best_model_info.model_type

model = best_model
loader = best_model_loader

### TRAINING
print("Beginning final training")
best_model.train()
optimizer = optim.Adam(best_model.parameters(), lr=1e-3)
NUM_EPOCHS = 10
train_losses = []
for epoch in range(NUM_EPOCHS):
    # print(f"epoch number {epoch}")

    total_loss = 0
    batch_num = 0

    for inputs, targets in train_out[best_model_loader]:
        # print(f"\tbatch number {batch_num}")
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # Forward pass, compute gradients, perform one training step.
        # Your code here!

        # Forward pass.
        output = best_model(inputs)

        # Compute loss.
        loss = loss_fn(output, targets)
        # print(f"\tloss: {loss}")

        optimizer.zero_grad()

        # Compute gradients based on the loss from the current batch (backpropagation).
        loss.backward()

        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()

        # Increment step counter
        total_loss += loss
        batch_num += 1

    train_loss = total_loss / batch_num

    # print(f"\ttraining loss: {train_loss}")
    train_losses.append(train_loss)

### FINAL TESTING
print(f"Final test on model {best_model_type}")
best_model.eval()
total_loss = 0
batch_num = 0
model_losses = []
with torch.no_grad():
    for inputs, targets in test_out[best_model_loader]:
        # print(f"Batch number {batch_num}")
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        output = best_model(inputs)
        loss = loss_fn(output, targets)
        total_loss += loss
        batch_num += 1

    average_loss = total_loss / batch_num
    model_losses.append(average_loss)

model_average = sum(model_losses) / len(model_losses)
print(f"Model {best_model_type} had an average loss of {model_average} in the final test")


# Save results
outer_train_loss_file = new_file_name("outer_train_losses", "csv")
pd.DataFrame([e.cpu().detach().numpy() for e in train_losses], columns=[f"{best_model_type} train loss"]).to_csv(outer_train_loss_file, index=False)
print(f"Wrote outer train loss to {outer_train_loss_file}")

outer_test_loss_file = new_file_name("outer_test_loss", "txt")
with open(outer_test_loss_file, "w") as f:
    f.write(str(model_average))
print(f"Wrote outer test loss to {outer_test_loss_file}")
