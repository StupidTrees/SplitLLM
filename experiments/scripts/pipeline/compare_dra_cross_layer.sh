# 实验：跨层黑盒攻击
seed=42

dataset_label='train'
model_name='gpt2-large'
exp_name='attacker_cross_different_sp'
client_num=1
global_round=1
client_steps=250
noise_scale=0.0
noise_mode="none"
attacker_dataset="codealpaca"
attacker_train_label='test'
attacker_prefix='normal'
attacker_search=False
data_shrink_frac=0.5
test_data_shrink_frac=0.5
evaluate_freq=500
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True
train_label='test'
test_label='test'
attack_model='gru'

# 观察不同的切分点
search_splits=(8 30)

# 观察不同的模型
datasets=('codealpaca')

for sp in "${search_splits[@]}"; do
  for dataset in "${datasets[@]}"; do
    # 先训练攻击模型
    echo "Running train_attacker.py with seed=$seed, dataset=$dataset, model=$attack_model"
    python ../py/train_attacker.py \
      --model_name "$model_name" \
      --seed "$seed" \
      --dataset "$dataset" \
      --attack_model "$attack_model" \
      --attack_mode 'b2tr' \
      --split_point_1 "$sp" \
      --split_point_2 999 \
      --dataset_train_label "$train_label" \
      --dataset_test_label "$test_label" \
      --save_checkpoint True \
      --log_to_wandb False

    # 将其用于攻击
    echo "Running evaluate_dra_cross_layer.py with seed=$seed, dataset=$dataset"
    python evaluate_dra_cross_layer.py \
      --noise_mode "$noise_mode" \
      --model_name "$model_name" \
      --global_round "$global_round" \
      --seed "$seed" \
      --dataset "$dataset" \
      --noise_scale_dxp "$noise_scale" \
      --exp_name "$exp_name" \
      --attacker_b2tr_sp "$sp" \
      --attacker_tr2t_sp "$sp" \
      --attacker_prefix "$attacker_prefix" \
      --attacker_search "$attacker_search" \
      --self_pt_enable "$self_pt_enable" \
      --client_num "$client_num" \
      --data_shrink_frac "$data_shrink_frac" \
      --test_data_shrink_frac "$test_data_shrink_frac" \
      --evaluate_freq "$evaluate_freq" \
      --client_steps "$client_steps" \
      --lora_at_top "$lora_at_top" \
      --lora_at_trunk "$lora_at_trunk" \
      --lora_at_bottom "$lora_at_bottom" \
      --collect_all_layers "$collect_all_layers" \
      --dataset_label "$dataset_label" \
      --attacker_dataset "$attacker_dataset" \
      --attacker_train_label "$attacker_train_label"
  done
done
