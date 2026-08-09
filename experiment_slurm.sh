#!/bin/bash
#SBATCH --job-name=pribavoj_quad_swarm      # job name
#SBATCH --output=experiment_name_%j.out # output file (%j = jobID)
#SBATCH --error=experiment_name_%j.err  # error file
#SBATCH --time=23:50:00                 # wall time limit
#SBATCH --partition=amdgpu             # or gpufast if you need GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1                  # uncomment if you need GPUs
echo "SLURM_JOBID = $SLURM_JOBID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

ml SciPy-bundle/2023.11-gfbf-2023b PyTorch/2.5.0-foss-2023b-CUDA-12.4.0 typing-extensions/4.11.0-GCCcore-13.2.0
echo "Modules loaded"
source ~/quad-swarm-rl/quad-swarm-env/bin/activate
echo "virtual environment sourced"
# Run your experiment
python -m swarm_rl.sb_train --num_env 15 "$@"
echo "Finished at: $(date)"