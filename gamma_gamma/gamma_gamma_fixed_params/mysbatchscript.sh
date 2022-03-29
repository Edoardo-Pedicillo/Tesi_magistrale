#!/bin/bash
              #SBATCH --cpus-per-task=3
              #SBATCH --mem-per-cpu=1G
              #SBATCH --time=5-0:0
	export NUMBA_NUM_THREADS=1
	/home/edoardopedicillo/miniconda3/bin/python3 /home/edoardopedicillo/gamma_gamma_fixed_params/quantum_classical_1Dgamma.py
