#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J DNN

### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4 
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB

### -- Select the resources: 1 gpu in exclusive process mode --
##BSUB -gpu "num=1:mode=exclusive_process"
##BSUB -R "select[gpu32gb]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
# BSUB -u
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o DNN.out 
#BSUB -e DNN.err 

# here follow the commands you want to execute with input.in as the input file
# bsub < ioannis_DNN_tissues_2.sh
# bstat
source ../venv_1/bin/activate
module load cuda/12.2.2

python DNN.py