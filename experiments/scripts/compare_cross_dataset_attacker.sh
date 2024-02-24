# 实验：跨层黑盒攻击
seed=42

dataset_label='train'
model_name='gpt2-large'
exp_name='attacker_cross_dataset'
client_num=1
global_round=1
client_steps=250
noise_scale=0.0
noise_mode="none"
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

attack_model='gru'
attacker_sp=23

attacker_datasets=( "codealpaca" "piqa" "dialogsum" "gsm8k" "wikitext")
# 观察不同的模型
sfl_datasets=("dialogsum" "gsm8k" "wikitext" "codealpaca" "piqa")

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    if [ "$attacker_dataset" == "$sfl_dataset" ]; then
      continue
    fi

    attacker_train_label='validation'
    attacker_test_label='test'
    if [ "$attacker_dataset" == "codealpaca" ] || [ "$attacker_dataset" == "gsm8k" ]; then
      attacker_train_label='test'
    fi


    # 先训练攻击模型
    echo "Running train_attacker.py with atk_ds=$attacker_dataset"
    python train_attacker.py \
      --model_name "$model_name" \
      --seed "$seed" \
      --dataset "$attacker_dataset" \
      --attack_model "$attack_model" \
      --attack_mode 'b2tr' \
      --split_point_1 "$attacker_sp" \
      --split_point_2 999 \
      --dataset_train_label "$attacker_train_label" \
      --dataset_test_label "$attacker_test_label" \
      --save_checkpoint True \
      --log_to_wandb False


    # 将其用于攻击
    echo "Running evaluate_attacker_cross_layer.py with sfl_ds=$sfl_dataset"
    python evaluate_attacker_cross_dataset.py \
      --noise_mode "$noise_mode" \
      --model_name "$model_name" \
      --global_round "$global_round" \
      --seed "$seed" \
      --dataset "$sfl_dataset" \
      --noise_scale "$noise_scale" \
      --exp_name "$exp_name" \
      --attacker_b2tr_sp "$attacker_sp" \
      --attacker_tr2t_sp "$attacker_sp" \
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
