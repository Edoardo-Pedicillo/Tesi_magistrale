#!/bin/bash
#SBATCH --job-name=zero
#SBATCH --ntasks=1
#SBATCH --time=10-0:0
#SBATCH --partition=general
#SBATCH --exclude=fermi,bose,chandra,bethe,hawking,higgs,salam,schroedinger,maxwell,cooper,haas,maisner

DIR=${SLURM_SUBMIT_DIR}/${SLURM_JOBID}
echo $DIR

if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  DIR=${SLURM_SUBMIT_DIR}/${SLURM_JOBID}_1
  mkdir $DIR
  echo "Saving data in ${DIR}..."
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "Saving data in ${DIR}..."
  mkdir $DIR
fi

python digits.py --layers 3 --latent_dim 3  --lr 5e-1 \
                 --lr_d 1e-2 --dataset 8x8Digits \
	         --folder $DIR --nqubits 6 --pixels 64 \
	         --training_samples 150 --batch_samples 64 \
		 --digit 0
