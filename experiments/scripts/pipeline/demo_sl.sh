sps="6-27"
global_round=2
batch_size=6
client_steps=300
client_num=3
evaluate_freq=200
max_global_step=-1
exp_name='Example-Split-Learning'
model_name='gpt2-large'
sfl_dataset="wikitext"
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
dataset_label='train'
data_shrink_frac=1.0
test_data_shrink_frac=0.3

echo "Running sim_with_attacker.py ..."
python ../py/sim_with_attacker.py \
  --case_name "Test" \
  --model_name "$model_name" \
  --split_points "$sps" \
  --global_round "$global_round" \
  --dataset "$sfl_dataset" \
  --exp_name "$exp_name" \
  --client_num "$client_num" \
  --data_shrink_frac "$data_shrink_frac" \
  --test_data_shrink_frac "$test_data_shrink_frac" \
  --evaluate_freq "$evaluate_freq" \
  --client_steps "$client_steps" \
  --lora_at_top "$lora_at_top" \
  --lora_at_trunk "$lora_at_trunk" \
  --lora_at_bottom "$lora_at_bottom" \
  --dataset_label "$dataset_label" \
  --batch_size "$batch_size" \
  --max_global_step "$max_global_step" \
  --tag_enable False \
  --gma_enable False \
  --gsma_enable False \
  --sma_enable False \
  --eia_enable False \
  --sip_enable False
