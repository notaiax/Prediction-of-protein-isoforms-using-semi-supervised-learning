# Instructions

To replicate our results, please run the notebook file Main-Notebook-Isoforms-Predictions.ipynb. This file combines the code from the following files used during development, which you can find in the other_files directory:

* PCA.py
* VAE_training.py
* VAE_100.py
* VAE_500.py
* tsv_to_hdf5.py
* baseline.py
* training.py

PCA.py applies a Principal Component Analysis to gtex gene dataset. VAE training trains a VAE using archs4 dataset. VAE_100 and VAE_500 applies the trained VAE to gtex gene dataset. tsv_to_hdf5 converts the created PCA.tsv, VAE_100.tsv and VAE_500.tsv from PCA.py, VAE_100.py and VAE_500.py to PCA.hdf5, VAE_100.hdf5 and VAE_500.hdf5. baseline.py creates the baseline model. Finally, training.py applies a regression model on the four datasets: gtex gene, PCA, VAE 100 and VAE 500 to predict isoform expression level, using gtex isoform dataset to calculate the MSE.

  There are other files that are also useful but not present in the main notebook file but useful for the main files mentioned above:

  * IsoDatasets.py
  * explore_data.ipynb
  * que.py
