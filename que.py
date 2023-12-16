import sys
import os
import subprocess

# USAGE: python que.py [script name] [job name] [cpu memory] ["GPU" if gpu wanted]
def print_usage():
    print("USAGE: python que.py [script name] [job name] [cpu memory] ['GPU' if gpu wanted]")

if __name__ == "__main__":
    args = sys.argv
    
    if len(args) < 2:
        print("Please provide the name of the script you want to run")
        print_usage()
        exit()
    script = args[1]
    if not os.path.isfile(script):
        print(f"File {script} not found")
        print_usage()
        exit()
    
    if len(args) < 3:
        print("Please provide a job name")
        print_usage()
        exit()
    job_name = args[2]

    if len(args) < 4:
        print("Please provide the amount of CPU memory you want")
        print_usage()
        exit()
    memory = args[3]

    if len(args) >= 5 and args[4] == "GPU":
        gpu = True
        gpu_args = """
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu \"num=1:mode=exclusive_process\"
#BSUB -R \"select[gpu32gb]\"
"""
    else:
        gpu = False

    queue = "gpuv100" if gpu else "hpc"
    
    bash = f"""#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q {queue}
### -- set the job Name -- 
#BSUB -J {job_name}

### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4 
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem={memory}GB]"
#BSUB -M {memory}GB
{gpu_args if gpu else ""}
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
#BSUB -o {job_name}.out 
#BSUB -e {job_name}.err 

source ../venv_1/bin/activate
module load cuda/12.2.2

python {script}"""
    
    print(f"Running {script} on queue {queue} with {memory} GB of CPU memory as job {job_name}")

    with open("python_queue_temp.sh", "w") as f:
        f.write(bash)
    
    os.system("bsub < python_queue_temp.sh")
    
    os.remove("python_queue_temp.sh")