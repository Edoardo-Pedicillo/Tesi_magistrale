#!/bin/bash
              #SBATCH --cpus-per-task=3
              #SBATCH --mem-per-cpu=1G
              #SBATCH --time=5-0:0
              #SBATCH --array=0-10:2
	export NUMBA_NUM_THREADS=1
	srun /home/edoardopedicillo/miniconda3/bin/python3 /home/edoardopedicillo/gamma_logistic/fixed_params2/quantum_classical_1Dgamma.py --n_epochs=2000 --lr=0.1 --nparams=$SLURM_ARRAY_TASK_ID
