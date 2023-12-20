import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib as plt
from matplotlib.pyplot import plot, savefig
import torch

gtex_gene_path = "/dtu-compute/datasets/iso_02456/gtex_gene_expression_norm_transposed.tsv.gz"


def load_data_chunk(filename, chunk_size):
    # Load a chunk of data from a gzipped TSV file.
    return pd.read_csv(filename, sep='\t', compression='gzip', chunksize=chunk_size, random)

X = next(load_data_chunk(gtex_gene_path, chunk_size=10, random)).iloc[:,:].to_numpy()
print(X.shape)

#X_np = X.iloc[:,1:].to_numpy()

N = X.shape[0]

for i in range(N):
    sample_path = "tensors/" + str(X[i,0])
    torch.save(torch.tensor(X[i,1:]), sample_path)
    print("sample path: ", sample_path, " saved.")


