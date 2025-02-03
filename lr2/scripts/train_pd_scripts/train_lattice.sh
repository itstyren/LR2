#!/bin/bash
env='Repu-Gen-Net'
scenario='Final_Final'
algo='PPO'
exp="D5R5_DE10_00_Random"
seed_max=1
network='lattice'
mixed_num=4

# agent number
env_dim=4
n_rollout_threads=16
num_env_steps=200000


# Set initial, final, and interval values for dilemma_S and dilemma_T
initial_S=-0.33
final_S=-0.33
interval_S=0.02

initial_T=1.30
final_T=1.30
interval_T=0.02

# Loop through dilemma_S_values and dilemma_T_values
dilemma_S=$initial_S
while (( $(echo "$dilemma_S <= $final_S" | bc -l) )); do
  dilemma_T=$initial_T
  while (( $(echo "$dilemma_T <= $final_T" | bc -l) )); do
    for seed in `seq ${seed_max}`; do
      echo "Running with seed=${seed}, dilemma_S=${dilemma_S}, dilemma_T=${dilemma_T}"
      CUDA_VISIBLE_DEVICES=3 python ../train/train_lattice.py \
      --env_dim ${env_dim}  --cuda --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --dilemma_S ${dilemma_S} --dilemma_T ${dilemma_T} \
      --user_name 'username' --episode_length 20 --n_rollout_threads ${n_rollout_threads} --train_freq 320 --num_env_steps ${num_env_steps}  \
      --video_interval  $(( n_rollout_threads > 100 ? 1 : 100 / n_rollout_threads ))   --log_interval $(( n_rollout_threads > 16 ? 1 : 16 / n_rollout_threads )) \
      --use_linear_lr_decay --repu_lr 0.01 --dilemma_lr 0.01 --freq_type 'step' --mini_batch 40 --seed ${seed} --repu_training_type 'all'\
      --dilemma_n_epochs 5 --repu_n_epochs 5 --use_linear_update  --repu_reward_coef 0.6 --compare_reward_pattern 'none' \
      --dilemma_ent_coef_fraction 1 --dilemma_ent_coef_initial_eps 0.1 --dilemma_ent_coef_final_eps 0.0 --repu_alpha 0.6 \
       --mse_frac 0.2 --repu_ent_coef_fraction 0.2 --repu_ent_coef_initial_eps 0.0 --repu_ent_coef_final_eps 0.00   \
       --normalize_advantage --random_repu --network_type ${network} --mixed_num ${mixed_num} --repu_reward_fraction 1 \
    
    done
    
    dilemma_T=$(echo "$dilemma_T + $interval_T" | bc)
  done
  dilemma_S=$(echo "$dilemma_S + $interval_S" | bc)
done