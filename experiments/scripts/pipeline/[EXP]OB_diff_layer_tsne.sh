# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[EXP]OB_diff_layer_tsne'
client_num=1
global_round=1
client_steps=500
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=200
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='llama2'

batch_size=1

sfl_datasets=("sensimarked")

for sfl_dataset in "${sfl_datasets[@]}"; do

  case_name="${model_name}-${sfl_dataset}"

  echo "Running draw.py with sfl_ds=$sfl_dataset"
  python ../py/observe_tsne_layers.py \
    --case_name "$case_name" \
    --model_name "$model_name" \
    --split_points "6-26" \
    --global_round "$global_round" \
    --seed "$seed" \
    --dataset "$sfl_dataset" \
    --exp_name "$exp_name" \
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
    --batch_size "$batch_size"

done
