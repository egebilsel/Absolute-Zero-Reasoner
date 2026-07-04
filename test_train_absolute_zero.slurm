#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J azr-test
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/Absolute-Zero-Reasoner
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --output=/dl_scratch2/ege/ege/repos/Absolute-Zero-Reasoner/azr_test-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/Absolute-Zero-Reasoner/azr_test-%j.err
#SBATCH --time=01:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/absolute-zero

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/verl"

python -m absolute_zero_reasoner.main_azr_ppo \
    data.shuffle=True \
    actor_rollout_ref.ref.include_ref=False \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/code_reason/test_answer.parquet \
    data.val_files=data/code_reason/test_answer.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=1312 \
    data.max_prompt_length=6144 \
    data.max_response_length=8096 \
    azr.data_selection_strategy.content_max_length=5600 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.pretrained_tokenizer=True \
    trainer.total_episodes=5 \
    trainer.save_freq=5 \
    trainer.default_local_dir=checkpoints/coder3b_test/ \
    trainer.logger=['console']
