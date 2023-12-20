import IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Here is an example of loading the Archs4 gene expression dataset and looping over it
# If you have about 12GB of memory, you can load the dataset to memory using the argument load_in_mem=True
archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")

archs4_train_dataloader = DataLoader(archs4_train, batch_size=64, shuffle=True)

for X in tqdm(archs4_train_dataloader):
    # You use this as part of your training loop
    pass

# Example of making a training set that excludes samples from the brain and a test set with only samples from the brain
# If you have enough memory, you can load the dataset to memory using the argument load_in_mem=True
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", include='brain')

print("gtex training set size:", len(gtex_train))
print("gtex test set size:", len(gtex_test))

gtx_train_dataloader = DataLoader(gtex_train, batch_size=64, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=64, shuffle=True)

for X,y in tqdm(gtx_train_dataloader):
    pass
