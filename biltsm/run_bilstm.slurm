#!/bin/bash

#SBATCH --job-name=meld_bilstm             # Job name
#SBATCH -A student                         # SLURM account (Required)
#SBATCH --output=job_output_%j.txt         # Standard output log
#SBATCH --error=job_error_%j.txt           # Error log
#SBATCH --partition=ucsx                   # Target GPU queue
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=4                  # CPU cores per task
#SBATCH --time=03:00:00                    # Time limit (hh:mm:ss)
#SBATCH --mem=16G                          # Memory allocation  

# Execute inside Singularity with --nv for CUDA/GPU access
srun singularity exec --nv meld_bilstm.sif python train_eval.py

# RUN THE SCRIPT ON THE COMPUTE NODE
# sbatch run_bilstm.slurm
