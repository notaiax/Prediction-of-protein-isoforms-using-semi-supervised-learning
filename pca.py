import pandas as pd
import numpy as np
from scipy.linalg import svd
from matplotlib.pyplot import plot, savefig

def load_data_chunk(filename, chunk_size):
    """ Load a chunk of data from a gzipped TSV file. """
    return pd.read_csv(filename, sep='\t', compression='gzip', chunksize=chunk_size)

gtex_gene_path = "/dtu-compute/datasets/iso_02456/gtex_gene_expression_norm_transposed.tsv.gz"


X = next(load_data_chunk(gtex_gene_path, chunk_size=1000)).iloc[1:, 1:].to_numpy()

#X = X[:,1:]
N = X.shape[0]
print(f"X.dtype: {X.dtype}")
print(f"X.shape: {X.shape}")
print(X[-1,:])

Y = X - np.ones((N, 1))*X.mean(0)
print(f"Y.dtype: {Y.dtype}")
print(f"Y.shape: {Y.shape}")
print(Y[-1,:])

stds = np.std(Y,0)
print(np.min(stds))
print(np.argmin(stds))

Y = Y*(1/np.std(Y,0))
print(f"Y.dtype: {Y.dtype}")
print(f"Y.shape: {Y.shape}")
print(Y[-1,:])
U,S,Vh = svd(Y,full_matrices=False)
V = Vh.T    
print(f"V.dtype: {V.dtype}")
print(f"V.shape: {V.shape}")
Z = Y @ V
print(f"Z.dtype: {Z.dtype}")
print(f"Z.shape: {Z.shape}")
print(Z)
plot(Z[:,0],Z[:,1], "b.")
savefig("plot.png")