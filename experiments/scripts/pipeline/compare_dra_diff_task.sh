# 实验：跨层黑盒攻击
seed=42

dataset_label='train'
model_name='bert'
exp_name='attacker_diff_task'
client_num=1
global_round=1
client_steps=250
noise_scale=0.0
noise_mode="none"
attacker_prefix='normal'
attacker_search=False
data_shrink_frac=0.09
test_data_label='test'
test_data_shrink_frac=0.02
evaluate_freq=250
attacker_freq=250
attacker_samples=30
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=False
lora_at_top=False
collect_all_layers=False

attack_model='gru'
attacker_sp=4
split_points='4-10'
datasets=("imdb")
task_types=("clsf")

for dataset in "${datasets[@]}"; do
  for task_type in "${task_types[@]}"; do
    attacker_train_label='test'
    attacker_test_label='unsupervised'
    attacker_train_frac=0.1
    attacker_test_frac=0.01
    # 先训练攻击模型
    echo "Running train_attacker.py with atk_ds=$dataset"
    python ../py/train_attacker.py \
      --model_name "$model_name" \
      --seed "$seed" \
      --dataset "$dataset" \
      --attack_model "$attack_model" \
      --attack_mode 'b2tr' \
      --split_point_1 "$attacker_sp" \
      --split_point_2 999 \
      --dataset_train_label "$attacker_train_label" \
      --dataset_test_label "$attacker_test_label" \
      --dataset_train_frac "$attacker_train_frac" \
      --dataset_test_frac "$attacker_test_frac" \
      --save_checkpoint True \
      --log_to_wandb False

    # 将其用于攻击
    echo "Running evaluate_attacker_diff_task.py with sfl_ds=$dataset"
    python ../py/evaluate_attacker_diff_task.py \
      --noise_mode "$noise_mode" \
      --model_name "$model_name" \
      --global_round "$global_round" \
      --seed "$seed" \
      --dataset "$dataset" \
      --noise_scale "$noise_scale" \
      --exp_name "$exp_name" \
      --attacker_b2tr_sp "$attacker_sp" \
      --attacker_tr2t_sp "$attacker_sp" \
      --attacker_prefix "$attacker_prefix" \
      --attacker_search "$attacker_search" \
      --self_pt_enable "$self_pt_enable" \
      --client_num "$client_num" \
      --data_shrink_frac "$data_shrink_frac" \
      --test_data_label "$test_data_label" \
      --test_data_shrink_frac "$test_data_shrink_frac" \
      --evaluate_freq "$evaluate_freq" \
      --client_steps "$client_steps" \
      --lora_at_top "$lora_at_top" \
      --lora_at_trunk "$lora_at_trunk" \
      --lora_at_bottom "$lora_at_bottom" \
      --collect_all_layers "$collect_all_layers" \
      --dataset_label "$dataset_label" \
      --attacker_dataset "$dataset" \
      --attacker_train_label "$attacker_train_label" \
      --split_points "$split_points" \
      --attacker_train_frac "$attacker_train_frac" \
      --task_type "$task_type" \
      --attacker_freq "$attacker_freq" \
      --attacker_samples "$attacker_samples"
  done
done
