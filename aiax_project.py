import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

def load_data_chunk(filename, chunk_size=1000):
    """ Load a chunk of data from a gzipped TSV file. """
    return pd.read_csv(filename, sep='\t', compression='gzip', chunksize=chunk_size)

# Define the file paths
archs4_path = "/dtu-compute/datasets/iso_02456/archs4_gene_expression_norm_transposed.tsv.gz"
gtex_gene_path = "/dtu-compute/datasets/iso_02456/gtex_gene_expression_norm_transposed.tsv.gz"
gtex_isoform_path = "/dtu-compute/datasets/iso_02456/gtex_isoform_expression_norm_transposed.tsv.gz"
gtex_anno_path = "/dtu-compute/datasets/iso_02456/gtex_gene_isoform_annoation.tsv.gz"
gtex_tissue_path = "/dtu-compute/datasets/iso_02456/gtex_annot.tsv.gz"


archs4_chunk = next(load_data_chunk(archs4_path, chunk_size=10))
gtex_gene_chunk = next(load_data_chunk(gtex_gene_path, chunk_size=10))
gtex_isoform_chunk = next(load_data_chunk(gtex_isoform_path, chunk_size=10))
gtex_anno_chunk = next(load_data_chunk(gtex_anno_path, chunk_size=10))
gtex_tissue_chunk = next(load_data_chunk(gtex_tissue_path, chunk_size=10))

# Example for one dataset
print(archs4_chunk.shape)
print(archs4_chunk.head())
archs4_chunk.describe()

# Histogram for a selected column in archs4_chunk
archs4_chunk.iloc[:, 1].hist(bins=50)
plt.title('Distribution of a Selected Gene Expression')
plt.xlabel('Expression Level')
plt.ylabel('Frequency')
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming the first column in gtex_gene_chunk is non-numeric data like sample identifiers
features = gtex_gene_chunk.columns[1:]

# Separating out the features (gene expression levels)
x = gtex_gene_chunk.loc[:, features].values

# Standardizing the features
x_standardized = StandardScaler().fit_transform(x)


pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_standardized)

# Creating a DataFrame for the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])


plt.figure(figsize=(8,6))
plt.scatter(principal_df['PC1'], principal_df['PC2'], s=50)
plt.title('PCA of GTEx Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig("fig.png")


print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2f}")
