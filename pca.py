import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib as plt
from matplotlib.pyplot import plot, savefig

gtex_gene_path = "/dtu-compute/datasets/iso_02456/gtex_gene_expression_norm_transposed.tsv.gz"


def load_data_chunk(filename, chunk_size):
    # Load a chunk of data from a gzipped TSV file.
    return pd.read_csv(filename, sep='\t', compression='gzip', chunksize=chunk_size, header=None)

X = next(load_data_chunk(gtex_gene_path, chunk_size=1000)).iloc[:, :].to_numpy()

"""
def load_data(filename):
    # Load a chunk of data from a gzipped TSV file.
    return pd.read_csv(filename, sep='\t', compression='gzip')

X = load_data(gtex_gene_path).iloc[:, 1:].to_numpy()
"""
print(X[0:2,0:2])
"""
N = X.shape[0]
print("--------------------------------------")

print(f"X.shape: {X.shape}")

Y = X - np.ones((N, 1))*X.mean(0)
print(f"Y.shape: {Y.shape}")

Y = Y*(1/np.std(Y,0))
print(f"Y.shape: {Y.shape}")

U,S,Vh = svd(X,full_matrices=False)
V = Vh.T    

print(f"V.shape: {V.shape}")
print(f"S.shape: {S.shape}")
print(f"U.shape: {U.shape}")

Z = Y @ V
print(f"Z.shape: {Z.shape}")

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Convert to pandas DataFrame
#df = pd.DataFrame(Z, columns=['Column1', 'Column2', 'Column3'])
Z_df = pd.DataFrame(Z)

# Save to CSV
# Save to TSV
Z_df.to_csv('Z.tsv.gz', sep='\t', index=False, compression="gzip")

rho_df = pd.DataFrame(rho)

# Save to CSV
# Save to TSV
rho_df.to_csv('rho.tsv.gz', sep='\t', index=False, compression="gzip")
"""
"""
print(Z)
plot(Z[:,0],Z[:,1], "b.")
savefig("plot.png")
"""

"""
# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
plt.savefig("variance.png")
"""