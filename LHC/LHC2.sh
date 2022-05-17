#!/bin/bash
#SBATCH --nodes=1 # request one node
#SBATCH --cpus-per-task=3  # ask for 1 cpu
#SBATCH --mem=1G # Maximum amount of memory this job will be given, try to estimate this to the best of your ability. This asks for 1 GB of ram.
#SBATCH --time=5-0:0 # ask that the job be allowed to run for 30 minutes.
#SBATCH --array=0-34:2 #specify how many times you want a job to run, we have a total of 7 array spaces

# everything below this line is optional, but are nice to have quality of life things
#SBATCH --output=job.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out
#SBATCH --error=job.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err
#SBATCH --jobname="example job" # a nice readable name to give your job so you know what it is when you see it in the queue, instead of just numbers

# under this line, we can load any modules if necessary

#below this line is where we can place our commands, in this case it will just simply output the task ID of the array
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
export NUMBA_NUM_THREADS=1
	srun /home/edoardopedicillo/miniconda3/bin/python3 /home/edoardopedicillo/tesi/LHC/quantum_classical_LHC_modified.py --n_epochs=5000 --lr=0.1 --n_params=$SLURM_ARRAY_TASK_ID --iterator=3
