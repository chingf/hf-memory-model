#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=theory      # The account name for the job.
#SBATCH --job-name=hf-ring       # The job name.
#SBATCH --mail-type=ALL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=chingfang17@gmail.com    # Where to send mail (e.g. uni123@columbia.edu)
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --time=30:00:00              # The time the job will take to run.
#SBATCH --output=array_%A-%a.log
#SBATCH --array=0-1
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.

FILENAMES=(overlap nonoverlap)

python run.py ${FILENAMES[$SLURM_ARRAY_TASK_ID]}

# End of script
