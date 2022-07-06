#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1G
#SBATCH --time=2-0:0
#SBATCH --exclude=fermi,bose,chandra,bethe,hawking,higgs,salam,schroedinger,maxwell,cooper,haas,maisner
#SBATCH --array=0-4:1 #specify how many times you want a job to run
export NUMBA_NUM_THREADS=1
srun python quantum_classical_2Dgamma.py --lr=0.1 --n_epochs=5000 --iterator=$SLURM_ARRAY_TASK_ID
