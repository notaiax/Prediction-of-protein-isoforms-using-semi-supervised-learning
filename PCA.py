###############################################################
# IMPORT PACKAGES
###############################################################

import pandas as pd
import numpy as np
from scipy.linalg import svd
import h5py

###############################################################
# IMPORT DATA
###############################################################

f_gtex_gene = h5py.File('/dtu-compute/datasets/iso_02456/hdf5/gtex_gene_expression_norm_transposed.hdf5', mode='r')
X = f_gtex_gene['expressions'][:]

row_header = list(f_gtex_gene['row_names'][:])
row_header = [row.decode('utf-8') for row in row_header]
row_header = np.array(row_header)
row_header = row_header.reshape(-1,1)

col_header = np.hstack((np.array(["id"]), np.arange(2000)))

###############################################################
# PCA
###############################################################

N = X.shape[0]

Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))

U,S,Vh = svd(X,full_matrices=False)
V = Vh.T    

Z = Y @ V
Z = Z[:,:2000]
print(f"Z.shape: {Z.shape}")

###############################################################
# REARRANGE DATA WITH HEADERS
###############################################################

Z = np.hstack((row_header, Z))

print(f"Z.shape: {Z.shape}")

###############################################################
# SAVE DATA
###############################################################

Z_df = pd.DataFrame(Z, columns = col_header)
Z_df.to_csv('PCA.tsv.gz', sep='\t', index=False, compression="gzip")

###############################################################
# VARIANCE
###############################################################

rho = (S*S) / (S*S).sum()

rho_df = pd.DataFrame(rho)
rho_df.to_csv('rho.tsv.gz', sep='\t', index=False, compression="gzip")