import IsoDatasets
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# Here is an example of doing PCA on arch4 using scikit-learn
#archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
archs4_train_dataloader = DataLoader(archs4_train, batch_size=64, shuffle=True)

ipca = IncrementalPCA(n_components=2)

for X in tqdm(archs4_train_dataloader):
    ipca.partial_fit(X.numpy())

