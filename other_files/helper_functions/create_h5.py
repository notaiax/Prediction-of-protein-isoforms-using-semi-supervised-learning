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

def add_tissue_hdf5(tissue_dict:dict, hdf5_file:h5py.File, n_rows:int, dataset_name='tissue'):
    dset = hdf5_file.create_dataset(dataset_name, (n_rows-1,), dtype=h5py.string_dtype())

    for i in range(n_rows-1):
        row_name = hdf5_file['row_names'][i].decode()
        dset[i] = tissue_dict[row_name]

if __name__ == "__main__":
    input_path = "/dtu-compute/datasets/iso_02456/"
    output_path = "/dtu-compute/datasets/iso_02456/hdf5/"

    #input_path = "../data/"
    #output_path = "../data/hdf5/"

    # Convert archs4_gene_expression to hdf5 format
    csv_filename = input_path + 'archs4_gene_expression_norm_transposed.tsv.gz'
    hdf5_filename = output_path + 'archs4_gene_expression_norm_transposed.hdf5'

    print("Counting rows and columns in:", csv_filename)
    #n_rows, n_cols = csv_count_rows_cols(csv_filename)
    n_rows, n_cols = 167885, 18966
    print(f"\t(n_rows, n_cols) = ({n_rows}, {n_cols})")

    f_archs4 = h5py.File(hdf5_filename, mode='w')
    print("Converting", csv_filename, "->", hdf5_filename)
    csv_to_hdf5(csv_filename, f_archs4, n_rows, n_cols)
    
    f_archs4.close()

    # Make a dict of the tissue types
    with gzip.open(input_path + 'gtex_annot.tsv.gz', 'rt') as file:
        tissue_dict = dict(csv.reader(file, delimiter='\t'))

    # Convert gtex_gene_expression to hdf5 format
    csv_filename = input_path + 'gtex_gene_expression_norm_transposed.tsv.gz'
    hdf5_filename = output_path + 'gtex_gene_expression_norm_transposed.hdf5'
    
    print("Counting rows and columns in:", csv_filename)
    #n_rows, n_cols = csv_count_rows_cols(csv_filename)
    n_rows, n_cols = (17357, 18966)
    print(f"\t(n_rows, n_cols) = ({n_rows}, {n_cols})")

    print("Converting", csv_filename, "->", hdf5_filename)
    f_gtex_gene = h5py.File(hdf5_filename, mode='w')
    csv_to_hdf5(csv_filename, f_gtex_gene, n_rows, n_cols)

    print("Adding tissue type to" , hdf5_filename)
    add_tissue_hdf5(tissue_dict, f_gtex_gene, n_rows)


    # Convert gtex_isoform_expression to hdf5 format
    csv_filename = input_path + 'gtex_isoform_expression_norm_transposed.tsv.gz'
    hdf5_filename = output_path + 'gtex_isoform_expression_norm_transposed.hdf5'
    
    print("Counting rows and columns in:", csv_filename)
    #n_rows, n_cols = csv_count_rows_cols(csv_filename)
    n_rows, n_cols = (17357, 156959)
    print(f"\t(n_rows, n_cols) = ({n_rows}, {n_cols})")
    
    print("Converting", csv_filename, "->", hdf5_filename)
    f_gtex_isoform = h5py.File(hdf5_filename, mode='w')
    csv_to_hdf5(csv_filename, f_gtex_isoform, n_rows, n_cols)
    add_tissue_hdf5(tissue_dict, f_gtex_isoform, n_rows)

    # Checking that gtex_gene and gtex_isoform is sorted the same way:
    sorted_equal = list(f_gtex_gene['row_names']) == list(f_gtex_isoform['row_names'])
    print("gtex_gene and gtex_isoform is sorted the same way: ", sorted_equal)

    f_gtex_gene.close()
    f_gtex_isoform.close()
