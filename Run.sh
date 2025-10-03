#!/bin/bash
#SBATCH --job-name=Tangled
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=opumni@myumanitoba.ca
#SBATCH --mail-type=END,FAIL

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4 
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00

mkdir -p logs

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

module load StdEnv/2023 cuda/12.2
module load python/3.11

cd /home/opumni/projects/def-shaiful/opumni
source venv/llm/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=<token>
export HF_HOME=$SLURM_TMPDIR/hf_cache

mkdir -p $HF_HOME

NAME=${1:-"openai/gpt-oss-20b"}

NAME=${1:-"openai/gpt-oss-20b"}
START=${2:-0}
END=${3:--1}
RQ=${4:-11}
echo "Using $NAME for job ID $SLURM_JOB_ID"

cd Tangled
srun python OpenSourceModels.py --job-id "$SLURM_JOB_ID" --name "$NAME" --start "$START" --end "$END" --rq "$RQ"
echo "Job finished at $(date)"
