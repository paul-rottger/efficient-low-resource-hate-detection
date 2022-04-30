#!/bin/sh

#SBATCH --job-name=test-finetune
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=short 
#SBATCH --output=outputs.out
#SBATCH --error=errors.err
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
