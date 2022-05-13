### CONNECTING TO ARC
Activate Oxford VPN
ssh -X sedm6193@arc-login.arc.ox.ac.uk
ssh -X sedm6193@htc-login.arc.ox.ac.uk

## CONDA ON ARC
module load Anaconda3/2020.11
source activate /data/engs-hatespeech/sedm6193/conda-envs/lrh-env
HINT: Deactivate venv BEFORE submitting a SLURM job using sbatch

### USEFUL SLURM COMMANDS
Show compute budget:    mybalance
Submit job:             sbatch script.sh
Cancel job:             scancel JOB_ID
Show queue status:      squeue
Show remaining storage: myquota

### SLURM CPU vs GPU
#SBATCH --ntasks-per-node=1
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1

### FILE TRANSFER
rsync -rP sedm6193@gateway.arc.ox.ac.uk:/data/engs-hatespeech/sedm6193/low-resource-hate/results/ "/Users/paul/Documents/Uni/PhD - Oxford/0 - Thesis/0_Articles/6_Low Resource Hate Speech Detection/low-resource-hate-speech-detection/3_evaluation/results"

