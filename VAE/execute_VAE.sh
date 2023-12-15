#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J training_gpu

### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4 
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"

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
#BSUB -o job_outputs/training_gpu.out 
#BSUB -e job_outputs/training_gpu.err 

# bsub < execute_VAE.sh
# bstat
# bkill job_ID

nvidia-smi
# Load the cuda module
source ../venv_iso/bin/activate
module swap cuda/12.2.2

python aiax_VAE_100_deeper_net.py
