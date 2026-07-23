#!/usr/bin/env bash
#SBATCH --job-name=yam-openpi-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-yam-infer-%j.out

set -euo pipefail

I2RT_ROOT="${I2RT_ROOT:?Set I2RT_ROOT to the i2rt_rllab checkout}"
echo "compute hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
exec "$I2RT_ROOT/policy_serving/launch_openpi_wire.sh"
