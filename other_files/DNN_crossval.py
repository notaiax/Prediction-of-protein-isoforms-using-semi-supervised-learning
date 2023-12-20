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
from dataclasses import dataclass
from typing import Callable, Any

def new_file_name(name: str, extension: str):
    num = 1
    files = {f for f in os.listdir('.') if os.path.isfile(f)}
    while f"{name}_{num}.{extension}" in files:
        num += 1
    
    return f"{name}_{num}.{extension}"

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
    #tissues_labels_dict = {idx:tissues_label for idx, tissues_label in zip(idxs, tissues_labels)}

    # Train / Test 1st level
    idx_train_test, idx_test = train_test_split(idx, test_size=int(len(idx)*test_size), stratify = tissue, random_state=1)
    tissue_train_test = [tissue[idx] for idx in idx_train_test]

    # Train / Test 2nd level
    stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    idx_train_list = []
    idx_test_list = []

    for idx_train, idx_test_inner in stratified_kfold.split(idx_train_test, tissue_train_test):
        
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

        idx_train_list.append(idx_train)
        idx_test_list.append(idx_test_inner)

    #Return
    return (idx_test, idx_train_list, idx_test_list, tissue)

# Load tissues
tissues_pd = pd.read_csv("tissues.tsv.gz", sep = "\t", compression = "gzip", header = None)
tissues = tissues_pd.values
tissues = np.squeeze(tissues, axis=1)
tissues = tissues.tolist()[1:]

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
    "Brain - Nucleus accumbens (basal ganglia)"]
                               
test_size = 0.2
num_folds = 6

idx_test, idx_train_list, idx_test_list, tissue = extract_ids(tissues, list_excluding_tissues, test_size, num_folds)

###############################################################
# CREATE CLASS DATASET
###############################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model, num_comp_pca):
        f_gtex_isoform = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_isoform_expression_norm_transposed.hdf5', mode='r')
        f_gtex_gene = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_pca = h5py.File('pca_data.hdf5', mode='r')
        f_gtex_VAE_100_10 = h5py.File("VAE_data_100.hdf5", mode="r")
        f_gtex_VAE_500_10 = h5py.File("VAE_data.hdf5", mode="r")
        f_gtex_VAE_100_100 = h5py.File("VAE_100_100.hdf5", mode="r")
        f_gtex_VAE_500_100 = h5py.File("VAE_500_100.hdf5", mode="r")

        self.dset_isoform = f_gtex_isoform['expressions']
        self.dset_gene = f_gtex_gene['expressions']
        self.dset_pca = f_gtex_pca['expressions'][:,:num_comp_pca]
        self.dset_VAE_100_10 = f_gtex_VAE_100_10['expressions']
        self.dset_VAE_500_10 = f_gtex_VAE_500_10['expressions']
        self.dset_VAE_100_100 = f_gtex_VAE_100_100['expressions']
        self.dset_VAE_500_100 = f_gtex_VAE_500_100['expressions']

        assert(self.dset_gene.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_pca.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_VAE_100_10.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_VAE_500_10.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_VAE_100_100.shape[0] == self.dset_isoform.shape[0])
        assert(self.dset_VAE_500_100.shape[0] == self.dset_isoform.shape[0])
        
        if model == "gene":
            self.x = self.dset_gene
        elif model == "pca":
            self.x = self.dset_pca
        elif model == "VAE_100_10":
            self.x = self.dset_VAE_100_10
        elif model == "VAE_500_10":
            self.x = self.dset_VAE_500_10
        elif model == "VAE_100_100":
            self.x = self.dset_VAE_100_100
        elif model == "VAE_500_100":
            self.x = self.dset_VAE_500_100

    def __len__(self):
        return self.dset_isoform.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.dset_isoform[idx]

###############################################################
# LOAD DATA
###############################################################

#models = ["gene","pca","VAE"]
models = ["pca","VAE_100_100"]
num_comp_pca = 100
#num_latent_features = 100

train_loaders = []
test_inner_loaders = []
test_outer_loaders = []

for model in models:

    train_loaders_fold = []
    test_inner_loaders_fold = []

    for fold in range(num_folds):
    
        full_data = Dataset(model, num_comp_pca)

        ###############################################################
        # DATASETS
        ###############################################################

        train_dataset = Subset(full_data, indices = idx_train_list[fold])
        test_dataset = Subset(full_data, indices = idx_test_list[fold])

        ###############################################################
        # DATALOADERS
        ###############################################################

        # Balance train_loader
        tissue_train = [tissue[idx] for idx in idx_train_list[fold]]
        tissue_train_tensor = torch.tensor(tissue_train)
        class_counts = torch.bincount(tissue_train_tensor)
        weights = 1.0 / class_counts[tissue_train]
        train_sampler = WeightedRandomSampler(weights, len(weights))

        # Dataloaders
        batch_size = 64

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        # Appending
        train_loaders_fold.append(train_loader)
        test_inner_loaders_fold.append(test_loader)
        
    test_dataset = Subset(full_data, indices = idx_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Appending

    train_loaders.append(train_loaders_fold)
    test_inner_loaders.append(test_inner_loaders_fold)
    test_outer_loaders.append(test_loader)

###############################################################
# TRAINING / VALIDATION
###############################################################

NUM_FEATURES = next(iter(train_loaders[0][0]))[0].shape[1]
NUM_OUTPUT = next(iter(train_loaders[0][0]))[1].shape[1]
print(f"NUM_FEATURES: {NUM_FEATURES}")
print(f"NUM_OUTPUT: {NUM_OUTPUT}")

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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    name: str
    create_model: Callable[[], Callable]
    requested_loader: int

model_infos = [
    ModelInfo("PCA", lambda: Net(100), 0),
    ModelInfo("VAE 100", lambda: Net(100), 1),
    ModelInfo("Baseline", lambda: BaselineModel(), 0)
]


@dataclass
class ModelResult:
    model: Any
    train_losses: list[Any]
    test_loss: Any

NUM_EPOCHS = 20

selected_model_results: dict[str, ModelResult] = {}
print("Beginning training")
for model_info in model_infos:
    model_type = model_info.name
    print(f"Using model {model_type}")
    requested_loader = model_info.requested_loader
    train_loss_fold = []
    valid_loss_fold = []

    model_results: list[ModelResult] = []
    for fold in range(num_folds):
        print(f"Fold number {fold}")
        model = model_info.create_model()

        model.to(DEVICE)
        print(model)

        loss_fn = nn.MSELoss()

        if model_type != "Baseline":
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        

        ### TRAINING / TESTING
        model.train()
        valid_losses = []

        i = 0

        ### TRAINING
        train_losses = []
        for epoch in range(NUM_EPOCHS):
            print(f"epoch number {epoch}")

            total_loss = 0
            batch_num = 0

            for inputs, targets in train_loaders[requested_loader][fold]:
                print(f"\tbatch number {batch_num}")
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # Forward pass, compute gradients, perform one training step.
                # Your code here!

                # Forward pass.
                output = model(inputs)

                # Compute loss.
                loss = loss_fn(output, targets)
                print(f"\tloss: {loss}")

                if model_type != "Baseline":
                    # Clean up gradients from the model.
                    optimizer.zero_grad()

                    # Compute gradients based on the loss from the current batch (backpropagation).
                    loss.backward()

                    # Take one optimizer step using the gradients computed in the previous step.
                    optimizer.step()

                # Increment step counter
                total_loss += loss
                batch_num += 1

            train_loss = total_loss / batch_num

            print(f"\ttraining loss: {train_loss}")
            train_losses.append(train_loss)
            
        if model_type != "Baseline":
            torch.save(model.state_dict(), "models_100epochssimple/" + str(model_type) + str("_") + str(fold) + str(".pth"))
        
        print("Finished training.")

        ### TESTING
        with torch.no_grad():
            model.eval()
            total_loss = 0
            batch_num = 0

            for inputs, targets in test_inner_loaders[requested_loader][fold]:
                print(f"\ttesting batch number {batch_num}")
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                output = model(inputs)
                loss = loss_fn(output, targets)
                print(f"\tloss: {loss}")
                batch_num += 1
                
                total_loss += loss
            
            test_loss = total_loss / batch_num
            
        
        model_results.append(ModelResult(model, train_losses, test_loss))
    
    result = min(model_results, key = lambda e: e.test_loss)
    
    selected_model_results[model_type] = result



# Plot results
plt.figure()
plt.ylim(0, 2)

for (name, result) in selected_model_results.items():
    plt.plot([e.cpu().detach().numpy() for e in result.train_losses], label=f"{name} train loss")

    print(f"Train losses: {result.train_losses}")
    print(f"Test loss: {result.test_loss}")

plt.legend()
plt.title("Losses")
plt.xlabel("Epoch")
plt.xticks(range(NUM_EPOCHS))
plt.ylabel("Mean squared error")
plt.show()

fname = new_file_name("losses", "png")
plt.savefig(fname, dpi=700)
print(f"Saved loss plot to {fname}")

print(model)