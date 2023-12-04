import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib as plt
from matplotlib.pyplot import plot, savefig

gtex_gene_path = "Z.tsv.gz"
X_pd = pd.read_csv(gtex_gene_path, sep = "\t", compression = "gzip")

id = "X_id.tsv.gz"
id_pd = pd.read_csv(id, sep = "\t", compression = "gzip")

print("X_pd:")
print(X_pd.shape)
print(X_pd)
print("id_pd:")
print(id_pd.shape)
print(id_pd)

all_data = np.hstack((id_pd.to_numpy(), X_pd.to_numpy()))
row_header = np.hstack((np.array(["id"]), np.arange(X_pd.shape[1])))
print("row_header")
print(row_header.shape)
print(row_header)

print("all_data:")
print(all_data.shape)
print(all_data)

final = pd.DataFrame(all_data, columns=row_header)
print("final:")
print(final.size)
print(final)

final.to_csv('pca_data.tsv.gz', sep='\t', index=False, compression="gzip")