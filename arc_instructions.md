### CONNECTING TO ARC
Activate Oxford VPN
ssh -X sedm6193@arc-login.arc.ox.ac.uk

## CONDA ON ARC
module load Anaconda3/2020.11
source activate /data/engs-hatespeech/sedm6193/conda-envs/low-resource-hate
HINT: Deactivate venv BEFORE submitting a SLURM job using sbatch

### USEFUL COMMANDS
Show compute budget:    mybalance
Submit job:             sbatch script.sh
Show queue status:      squeue
Show remaining storage: myquota


