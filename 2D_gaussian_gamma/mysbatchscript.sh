#!/bin/bash
              #SBATCH --cpus-per-task=3
              #SBATCH --mem-per-cpu=1G
              #SBATCH --time=5-0:0
              #SBATCH --array=0-12:2
	echo " ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
	export NUMBA_NUM_THREADS=1
	srun /home/edoardopedicillo/miniconda3/bin/python3 /home/edoardopedicillo/tesi/2D_gaussian_gamma/quantum_classical_2Dgamma.py --n_epochs=5000 --lr=0.1 --n_params=$SLURM_ARRAY_TASK_ID
