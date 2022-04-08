#!/bin/bash
              #SBATCH --cpus-per-task=3
              #SBATCH --mem-per-cpu=1G
              #SBATCH --time=5-0:0
	export NUMBA_NUM_THREADS=1

	for i in 10000 5000 2000 1000 500 250 100
	do
	/home/edoardopedicillo/miniconda3/bin/python3 /home/edoardopedicillo/tesi/logistic_10_params/quantum_classical_logistic.py --training_samples i
done
