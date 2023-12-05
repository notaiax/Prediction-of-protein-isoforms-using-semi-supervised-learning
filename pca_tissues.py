import pandas as pd
import numpy as np
import h5py
import torch

class PCADataset(torch.utils.data.Dataset):
    def __init__(self):
        f_gtex_isoform = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        self.dset_isoform = f_gtex_isoform['expressions']
        self.tissues = list(f_gtex_isoform['tissue'][:])

    def __len__(self):
        return self.dset_isoform.shape[0]
        

    def __getitem__(self, idx):
        return self.dset_isoform[idx]
    
full_data = PCADataset()

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

print(len(tissues_labels))
print(tissues_labels)

tissues = np.array(tissues)
tissues_labels = np.array(tissues)

print(tissues.shape)
print(tissues)
print(tissues_labels.shape)
print(tissues_labels)

tissues_df = pd.DataFrame(tissues)
tissues_df.to_csv('tissues.tsv.gz', sep='\t', index=False, compression="gzip")

tissues_labels_df = pd.DataFrame(tissues_labels)
tissues_labels_df.to_csv('tissues_labels.tsv.gz', sep='\t', index=False, compression="gzip")




