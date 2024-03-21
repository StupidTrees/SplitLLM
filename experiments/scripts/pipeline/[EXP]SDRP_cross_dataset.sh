# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[EXP]SDRP_cross_dataset'
client_num=1
global_round=1
client_steps=500
noise_scale=0.0
noise_mode="none"
attacker_prefix='normal'
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=500
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='llama2'
attack_model='gru'
sps='6-26'
attacker_sp=6
batch_size=1

attacker_datasets=("sensimasked")
# 观察不同的模型
sfl_datasets=("codealpaca" "piqa" "dialogsum" "gsm8k" "wikitext")

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do

    # 先训练攻击模型
    echo "Running train_attacker.py with atk_ds=$attacker_dataset"
    python ../py/train_attacker.py \
      --model_name "$model_name" \
      --seed "$seed" \
      --dataset "$attacker_dataset" \
      --attack_model "$attack_model" \
      --attack_mode 'b2tr' \
      --noise_mode "$noise_mode" \
      --sps "$sps" \
      --save_checkpoint True \
      --log_to_wandb False

    case_name="${model_name}-${sfl_dataset}<${attacker_dataset}"

    # 将其用于攻击
    echo "Running evaluate_dra_cross_dataset.py with sfl_ds=$sfl_dataset"
    python ../py/evaluate_dra_cross_dataset.py \
      --noise_mode "$noise_mode" \
      --case_name "$case_name" \
      --model_name "$model_name" \
      --split_points "$sps" \
      --global_round "$global_round" \
      --seed "$seed" \
      --dataset "$sfl_dataset" \
      --noise_scale_dxp "$noise_scale" \
      --exp_name "$exp_name" \
      --attacker_b2tr_sp "$attacker_sp" \
      --attacker_tr2t_sp "$attacker_sp" \
      --attacker_prefix "$attacker_prefix" \
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
      --batch_size "$batch_size"
  done
done
