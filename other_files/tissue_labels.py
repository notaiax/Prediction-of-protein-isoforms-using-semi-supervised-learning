import pandas as pd
import torch

gtex_tissue_path = "/dtu-compute/datasets/iso_02456/gtex_annot.tsv.gz"
# Step 1: Read the TSV file
df = pd.read_csv(gtex_tissue_path, sep = "\t", compression = "gzip", header = None).iloc[1:]
#df = pd.read_csv('data.tsv', delimiter='\t', compressheader=None)
ids = df[0]
tissues = df[1]

# Step 2: Create a Mapping for Tissues to Numbers
tissue_to_number = {tissue: i for i, tissue in enumerate(tissues.unique())}

# Step 3: Create the Final Dictionary
id_to_tissue_number = {id_: tissue_to_number[tissue] for id_, tissue in zip(ids, tissues)}

gtex_gene_path = "/dtu-compute/datasets/iso_02456/gtex_gene_expression_norm_transposed.tsv.gz"
# Step 1: Read the TSV file
df = pd.read_csv(gtex_gene_path, sep = "\t", compression = "gzip", header = None).iloc[1:]

tissue_labels_list = [id_to_tissue_number[id] for id in df[0]]

#print(tissue_labels_list)
# Print the dictionary (optional)
#print("---------------------------------------------------------")
#print(id_to_tissue_number)
"""
for i, (key, value) in enumerate(id_to_tissue_number.items()):
    if i < 5:
        print(key, value)
    else:
        break
"""
torch.save(tissue_labels_list, 'tissue_dict_list.pth')