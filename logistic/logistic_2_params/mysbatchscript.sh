#!/bin/bash
#SBATCH --job-name=GAN
#SBATCH --ntasks=1
#SBATCH --time=10-0:0
#SBATCH --partition=general
#SBATCH --exclude=fermi,bose,chandra,bethe,hawking,higgs,salam,schroedinger,maxwell,cooper,haas,maisner
export NUMBA_NUM_THREADS=1


python  /home/edoardopedicillo/tesi/logistic/logistic_2_params/quantum_classical_1Dgamma.py --iterator=3
