#!/bin/sh

#SBATCH --job-name=test-finetune
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:10:00
#SBATCH --partition=short 
#SBATCH --output=test-finetune.out
#SBATCH --error=test-finetune.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk

# reset modules
module purge

# load python module
module load Anaconda3/2020.11

# activate the right conda environment
source activate $DATA/conda-envs/low-resource-hate

# Useful job diagnostics
#
nvidia-smi
#

python test.py 
