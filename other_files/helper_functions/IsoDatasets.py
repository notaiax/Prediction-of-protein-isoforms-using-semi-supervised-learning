import h5py
import re
import numpy as np
import torch.utils.data

class Archs4GeneExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, load_in_mem:bool=False):
        f_archs4 = h5py.File(data_dir + '/archs4_gene_expression_norm_transposed.hdf5', mode='r')
        self.dset = f_archs4['expressions']

        if load_in_mem:
            self.dset = np.array(self.dset)

    def __len__(self):
        return self.dset.shape[0]

    def __getitem__(self, idx):
        return self.dset[idx]


class GtexDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, include:str="", exclude:str="", load_in_mem:bool=False):
        f_gtex_gene = h5py.File(data_dir + 'gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_isoform = h5py.File(data_dir + 'gtex_isoform_expression_norm_transposed.hdf5', mode='r')

        self.dset_gene = f_gtex_gene['expressions']
        self.dset_isoform = f_gtex_isoform['expressions']

        assert(self.dset_gene.shape[0] == self.dset_isoform.shape[0])

        if load_in_mem:
            self.dset_gene = np.array(self.dset_gene)
            self.dset_isoform = np.array(self.dset_isoform)

        self.idxs = None

        if include and exclude:
            raise ValueError("You can only give either the 'include_only' or the 'exclude_only' argument.")

        if include:
            matches = [bool(re.search(include, s.decode(), re.IGNORECASE)) for s in f_gtex_gene['tissue']]
            self.idxs = np.where(matches)[0]

        elif exclude:
            matches = [not(bool(re.search(exclude, s.decode(), re.IGNORECASE))) for s in f_gtex_gene['tissue']]
            self.idxs = np.where(matches)[0]

    def __len__(self):
        if self.idxs is None:
            return self.dset_gene.shape[0]
        else:
            return self.idxs.shape[0]

    def __getitem__(self, idx):
        if self.idxs is None:
            return self.dset_gene[idx], self.dset_isoform[idx]
        else:
            return self.dset_gene[self.idxs[idx]], self.dset_isoform[self.idxs[idx]]
