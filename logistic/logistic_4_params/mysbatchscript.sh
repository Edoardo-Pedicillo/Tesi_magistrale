#!/bin/bash
#SBATCH --job-name=zero
#SBATCH --ntasks=1
#SBATCH --time=10-0:0
#SBATCH --partition=general
#SBATCH --exclude=fermi,bose,chandra,bethe,hawking,higgs,salam,schroedinger,maxwell,cooper,haas,maisner

#SBATCH --array=0-3:1 #specify how many times you want a job to run, we have a total of 7 array spaces
	
export NUMBA_NUM_THREADS=1
	
srun python /home/edoardopedicillo/tesi/logistic/logistic_4_params/quantum_classical_1Dgamma.py --iterator=$SLURM_ARRAY_TASK_ID

