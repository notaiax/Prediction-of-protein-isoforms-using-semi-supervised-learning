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

log_fname = new_file_name("log_test", "txt")

def print(msg):
    with open(log_fname, "a") as file:
        file.write(str(msg) + "\n")

print("Beginning log")

###############################################################
# EXTRACT IDXS TRAINING / TESTING
###############################################################

# Function
 
def extract_ids(tissues, test_size):
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
    idx_train, idx_test = train_test_split(idx, test_size=int(len(idx)*test_size), stratify = tissue, random_state=1)

    #Return
    return (idx_train, idx_test, tissue)

# Load tissues
tissues_pd = pd.read_csv("tissues.tsv.gz", sep = "\t", compression = "gzip", header = None)
tissues = tissues_pd.values
tissues = np.squeeze(tissues, axis=1)
tissues = tissues.tolist()[1:]
                               
test_size = 0.2

idx_train, idx_test, tissue = extract_ids(tissues, test_size)

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
models = ["gene","pca","VAE_100_100","VAE_500_100"]
num_comp_pca = 100
#num_latent_features = 100

train_loaders = []
test_loaders = []

for model in models:
    
    full_data = Dataset(model, num_comp_pca)

    ###############################################################
    # DATASETS
    ###############################################################

    train_dataset = Subset(full_data, indices = idx_train)
    test_dataset = Subset(full_data, indices = idx_test)

    ###############################################################
    # DATALOADERS
    ###############################################################

    # Balance train_loader
    tissue_train = [tissue[idx] for idx in idx_train]
    tissue_train_tensor = torch.tensor(tissue_train)
    class_counts = torch.bincount(tissue_train_tensor)
    weights = 1.0 / class_counts[tissue_train]
    train_sampler = WeightedRandomSampler(weights, len(weights))

    # Dataloaders
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Appending
    train_loaders.append(train_loader)
    test_loaders.append(test_loader)

###############################################################
# TRAINING / VALIDATION
###############################################################

NUM_OUTPUT = next(iter(train_loaders[0]))[1].shape[1]
print(f"NUM_OUTPUT: {NUM_OUTPUT}")

loss_fn = nn.MSELoss()

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


@dataclass
class ModelInfo:
    name: str
    create_model: Callable[[], Callable]
    requested_loader: int

model_infos = [
    ModelInfo("Original", lambda: Net(18965), 0),
    ModelInfo("PCA", lambda: Net(100), 1),
    ModelInfo("VAE 100", lambda: Net(100), 2),
    ModelInfo("VAE 500", lambda: Net(500), 3)
]

@dataclass
class LoadedModelInfo:
    name: str
    num: int
    model: Callable
    requested_loader: int

loaded_model_infos: list[LoadedModelInfo] = []
for info in model_infos:
    for i in range(6):
        model = info.create_model()
        model.to(DEVICE)

        # Load the state dictionary
        state_dict = torch.load("models_100epochs/" + info.name + "_" + str(i) + ".pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        loaded_model_infos.append(LoadedModelInfo(info.name, i, model, info.requested_loader))

### TESTING
model_names_averages: list[tuple[str, int, Any]] = []
print("Starting testing")
for info in loaded_model_infos:
    model_name = info.name
    model_num = info.num
    model = info.model
    loader = info.requested_loader

    print(f"Testing model {model_name}_{model_num}")
    model.eval()
    total_loss = 0
    batch_num = 0
    model_losses = []
    with torch.no_grad():
        for inputs, targets in test_loaders[loader]:
            # print(f"Batch number {batch_num}")
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            output = model(inputs)
            loss = loss_fn(output, targets)
            total_loss += loss
            batch_num += 1

        average_loss = total_loss / batch_num
        model_losses.append(average_loss)
    
    model_average = sum(model_losses) / len(model_losses)
    print(f"Model {model_name}_{model_num} had an average loss of {model_average}")
    res = (model_name, model_num, model_average)
    model_names_averages.append(res)

print("Creating bar plot")
plt.figure()

colors = ["b"] * 6 + ["g"] * 6 + ["r"] * 6 + ["c"] * 6
plt.bar([x[0] + f"_{x[1]}" for x in model_names_averages], [x[2] for x in model_names_averages], colors=colors)
plt.title("Model losses")
plt.xlabel("Model")
plt.ylabel("Mean squared error")
plt.show()

fname = new_file_name("test_losses_bar", "png")
plt.savefig(fname, dpi=700)
print(f"Saved bar plot to {fname}")


### SELECTION
best_model_name, best_model_num, best_average = min(model_names_averages, key=lambda tup: tup[1])
print(f"Model {best_model_name}_{best_model_num} had the best performance with an average loss of {best_average}")


for info in model_infos:
    if info.name == best_model_name:
        best_model = info.create_model()
        best_model_loader = info.requested_loader
        break
else:
    raise RuntimeError("Could not find model in list")

model = best_model
loader = best_model_loader

### TRAINING
print("Beginning training")
best_model.train()
optimizer = optim.Adam(best_model.parameters(), lr=1e-3)
NUM_EPOCHS = 10
train_losses = []
for epoch in range(NUM_EPOCHS):
    print(f"epoch number {epoch}")

    total_loss = 0
    batch_num = 0

    for inputs, targets in train_loaders[best_model_loader]:
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

    print(f"\ttraining loss: {train_loss}")
    train_losses.append(train_loss)

# Plot results
print("Creating final train plot")
plt.figure()
plt.ylim(0, 2)

plt.plot([e.cpu().detach().numpy() for e in train_losses], label=f"{best_model_name} train loss")

print(f"Final train losses: {train_losses}")

plt.legend()
plt.title("Final train losses")
plt.xlabel("Epoch")
plt.xticks(range(NUM_EPOCHS))
plt.ylabel("Mean squared error")
plt.show()

fname = new_file_name("final_train_losses", "png")
plt.savefig(fname, dpi=700)
print(f"Saved loss plot to {fname}")

### FINAL TESTING
print(f"Final test on model {best_model_name}")
best_model.eval()
total_loss = 0
batch_num = 0
model_losses = []
with torch.no_grad():
    for inputs, targets in test_loaders[best_model_loader]:
        print(f"Batch number {batch_num}")
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        output = best_model(inputs)
        loss = loss_fn(output, targets)
        total_loss += loss
        batch_num += 1

    average_loss = total_loss / batch_num
    model_losses.append(average_loss)

model_average = sum(model_losses) / len(model_losses)
print(f"Model {best_model_name} had an average loss of {model_average} in the final test")
