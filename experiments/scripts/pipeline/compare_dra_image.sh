# 实验：跨层黑盒攻击
seed=42

dataset_label='train'
model_name='vit'
exp_name='[TEST]Evaluate_DRA_Image'
client_num=1
global_round=10
client_steps=250

attacker_dataset="imagewoof"
attacker_prefix='normal'
attacker_model='vit'
data_shrink_frac=1.0
test_data_shrink_frac=0.5
evaluate_freq=200
self_pt_enable=False
lora_at_trunk=False
lora_at_bottom=False
lora_at_top=False
collect_all_layers=True
attack_model='vit'
max_global_step=2400
batch_size=32

datasets=('imagewoof')

search_splits=(6)

for sp in "${search_splits[@]}"; do
  for dataset in "${datasets[@]}"; do
    sps="$sp-10"
    # 先训练攻击模型
    echo "Running train_attacker_image.py with seed=$seed, dataset=$dataset, model=$attack_model"
    python ../py/train_attacker_image.py \
      --model_name "$model_name" \
      --seed "$seed" \
      --dataset "$dataset" \
      --attack_model "$attack_model" \
      --attack_mode 'b2tr' \
      --sps "$sps" \
      --save_checkpoint True \
      --log_to_wandb False

    # 将其用于攻击
    echo "Running evaluate_dra_image.py with seed=$seed, dataset=$dataset"
    python ../py/evaluate_dra_image.py \
      --model_name "$model_name" \
      --global_round "$global_round" \
      --seed "$seed" \
      --split_points "$sps" \
      --dataset "$dataset" \
      --exp_name "$exp_name" \
      --attacker_b2tr_sp "$sp" \
      --attacker_tr2t_sp "$sp" \
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
      --max_global_step "$max_global_step" \
      --batch_size "$batch_size"\
      --attacker_model "$attacker_model"\
      --test_data_label 'validation'
  done
done
