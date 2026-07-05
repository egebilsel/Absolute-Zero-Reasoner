#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J azr-test
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/Absolute-Zero-Reasoner
#SBATCH --nodelist=romer3
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --output=/dl_scratch2/ege/ege/repos/Absolute-Zero-Reasoner/azr_test-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/Absolute-Zero-Reasoner/azr_test-%j.err
#SBATCH --time=01:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/absolute-zero

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

bash scripts/selfplay/coder3b.sh \
    data.max_prompt_length=16384 \
    azr.data_selection_strategy.content_max_length=16384 \
    trainer.total_epochs=1 \
    trainer.logger=['console'] \
    trainer.default_local_dir=checkpoints/coder3b_test/
