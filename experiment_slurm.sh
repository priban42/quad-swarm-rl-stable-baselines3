#!/bin/bash
#SBATCH --job-name=pribavoj_quad_swarm      # job name
#SBATCH --output=experiment_name_%j.out # output file (%j = jobID)
#SBATCH --error=experiment_name_%j.err  # error file
#SBATCH --time=20:00:00                 # wall time limit
#SBATCH --partition=amdgpu             # or gpufast if you need GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=33
#SBATCH --gres=gpu:1                  # uncomment if you need GPUs

# Run your experiment
python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_modified --max_parallel=1 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1 &
