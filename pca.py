import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib as plt
from matplotlib.pyplot import plot, savefig

gtex_gene_path = "/dtu-compute/datasets/iso_02456/gtex_gene_expression_norm_transposed.tsv.gz"

X_pd = pd.read_csv(gtex_gene_path, sep = "\t", compression = "gzip", header = None)
X_np = X_pd.values
X_id = X_np[1:,0]
X_header = ["id"] + list(range(1, 2001))
X = X_np[1:,1:]
X = X.astype(float)

N = X.shape[0]

Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))

U,S,Vh = svd(X,full_matrices=False)
V = Vh.T    

Z = Y @ V
Z = Z[:,:2000]
print(f"Z.shape: {Z.shape}")

Z_df = pd.DataFrame(Z)
Z_df.to_csv('Z.tsv.gz', sep='\t', index=False, compression="gzip")

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()
print(rho[:2000])

# Concatenate X_id to the first column of X
X_combined_1 = np.concatenate((X_id, X), axis=1)
X_combined_2 = np.concatenate((X_header, X_combined_1), axis=0)
print(f"X_combined_2.shape: {X_combined_2.shape}")

# Convert to pandas DataFrame and save as tsv
X_df = pd.DataFrame(X_combined_2)
X_df.to_csv('X.tsv.gz', sep='\t', index=False, compression="gzip")

rho_df = pd.DataFrame(rho)
rho_df.to_csv('rho.tsv.gz', sep='\t', index=False, compression="gzip")