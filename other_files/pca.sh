#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J pca_iso
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=40GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 40GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s223237@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o pca_iso.out 
#BSUB -e pca_iso.err 

# here follow the commands you want to execute with input.in as the input file
# bsub < pca.sh
# bstat
source .venv/bin/activate
module load python3/3.11.4
module load pandas/2.0.2-python-3.11.4
module load numpy/1.24.3-python-3.11.4-openblas-0.3.23
module load matplotlib/3.7.1-numpy-1.24.3-python-3.11.4
module load scipy/1.10.1-python-3.11.4

python pca.py