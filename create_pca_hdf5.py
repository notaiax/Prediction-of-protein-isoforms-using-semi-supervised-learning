import csv
import gzip
import h5py
import numpy as np
from tqdm import tqdm

def csv_count_rows_cols(csv_filename:str) -> (int, int):
    n_rows = 0
    n_cols = -1

    with gzip.open(csv_filename, 'rt') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            n_rows += 1
            if n_cols < 0:
                n_cols = len(line)
            else:
                assert(n_cols == len(line))

    return n_rows, n_cols

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
    input_path = "/dtu-compute/datasets/iso_02456/"
    output_path = "/dtu-compute/datasets/iso_02456/hdf5/"

    # Convert archs4_gene_expression to hdf5 format
    csv_filename = 'pca_data.tsv.gz'
    hdf5_filename = 'pca_data.hdf5'

    print("Counting rows and columns in:", csv_filename)
    #n_rows, n_cols = csv_count_rows_cols(csv_filename)
    n_rows, n_cols = 17357, 2001
    print(f"\t(n_rows, n_cols) = ({n_rows}, {n_cols})")

    f_pca = h5py.File(hdf5_filename, mode='w')
    print("Converting", csv_filename, "->", hdf5_filename)
    csv_to_hdf5(csv_filename, f_pca, n_rows, n_cols)

    f_pca.close()

