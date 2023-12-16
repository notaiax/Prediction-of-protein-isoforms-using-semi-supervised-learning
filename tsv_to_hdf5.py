import csv
import gzip
import h5py
import numpy as np
from tqdm import tqdm

def csv_to_hdf5(csv_filename:str, hdf5_file:h5py.File, n_rows:int, n_cols:int, dataset_name='expressions', disable_progressbar=False):
    dset = hdf5_file.create_dataset(dataset_name, (n_rows-1, n_cols-1), dtype='float32')
    dset_row_names = hdf5_file.create_dataset('row_names', (n_rows-1,), dtype=h5py.string_dtype())
    dset_col_names = hdf5_file.create_dataset('col_names', (n_cols-1,), dtype=h5py.string_dtype())

    with gzip.open(csv_filename, 'rt') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in tqdm(enumerate(reader), total=n_rows, disable=disable_progressbar):
            if i == 0:
                dset_col_names[:] = line[1:]
            if i > 0:
                dset[i-1, :] = np.array(line[1:], dtype=float)
                dset_row_names[i-1] = line[0]

        assert(i==n_rows-1)

if __name__ == "__main__":

    models =  ["PCA","VAE_100","VAE_500"]
    n_rows = 17357
    n_cols = [2001, 101, 501]

    for model, n_col in zip(models, n_cols):

        # Convert archs4_gene_expression to hdf5 format
        csv_filename = "datasets_reduced/" + model + ".tsv.gz"
        hdf5_filename = "datasets_reduced/" + model + ".hdf5"

        print("Counting rows and columns in:", csv_filename)
        print(f"\t(n_rows, n_cols) = ({n_rows}, {n_col})")

        f_pca = h5py.File(hdf5_filename, mode='w')
        print("Converting", csv_filename, "->", hdf5_filename)
        csv_to_hdf5(csv_filename, f_pca, n_rows, n_col)

        f_pca.close()

