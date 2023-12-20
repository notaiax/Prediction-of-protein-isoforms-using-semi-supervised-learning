import h5py
import numpy as np

def sort_rows(input_hdf5_file, output_hdf5_file):
    permutation = np.argsort(input_hdf5_file['col_names'])

    for dataset_name in input_hdf5_file.keys():
        print("Copying:", dataset_name)
        if dataset_name == 'col_names':
            output_hdf5_file.create_dataset(dataset_name, data=input_hdf5_file[dataset_name][:][permutation])
        elif dataset_name == 'expressions':
            output_hdf5_file.create_dataset(dataset_name, data=input_hdf5_file[dataset_name][:][:,permutation])
            pass
        else:
            output_hdf5_file.create_dataset(dataset_name, data=input_hdf5_file[dataset_name])

def check_same_row_expression(input_hdf5_file, output_hdf5_file, gene_name):
    input_idx = np.where(input_hdf5_file['col_names'][:] == gene_name)[0][0]
    output_idx = np.where(output_hdf5_file['col_names'][:] == gene_name)[0][0]
    same_row_expression = np.all(input_hdf5_file['expressions'][:,input_idx] == output_hdf5_file['expressions'][:,output_idx])
    return same_row_expression

if __name__ == "__main__":

    input_path = "../hdf5/"
    output_path = "../hdf5-row-sorted/"

    input_file_archs4 = h5py.File(input_path + 'archs4_gene_expression_norm_transposed.hdf5', 'r')
    output_file_archs4 = h5py.File(output_path + 'archs4_gene_expression_norm_transposed.hdf5', 'w')
    sort_rows(input_file_archs4, output_file_archs4)

    input_file_gtex_gene = h5py.File(input_path + 'gtex_gene_expression_norm_transposed.hdf5', 'r')
    output_file_gtex_gene = h5py.File(output_path + 'gtex_gene_expression_norm_transposed.hdf5', 'w')
    sort_rows(input_file_gtex_gene, output_file_gtex_gene)

    # Check that the col names are sorted the same way
    col_sorted = output_file_archs4['col_names'][:] == output_file_gtex_gene['col_names'][:]
    print("Row names sorted the same way in the two files:", np.all(col_sorted))

    # Check that some random columns have the same values before and after sorting
    for i in np.random.randint(input_file_archs4['col_names'].shape[0], size=10):
        gene_name = input_file_archs4['col_names'][i]
        print("Gene:", gene_name.decode())
        archs4_same_row_expression = check_same_row_expression(input_file_archs4, output_file_archs4, gene_name)
        gtex_same_row_expression = check_same_row_expression(input_file_gtex_gene, output_file_gtex_gene, gene_name)
        print("\tSame row expression archs4:", archs4_same_row_expression)
        print("\tSame row expression gtex:", gtex_same_row_expression)