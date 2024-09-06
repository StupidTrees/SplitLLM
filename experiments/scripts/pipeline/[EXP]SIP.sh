# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[EXP]SIP'
global_round=1
client_steps=500
noise_scale=0.0
noise_mode="none"
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=900
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

sps="6-27"
batch_size=2

attacker_freq=200
attacker_samples=5
max_global_step=605

sip_inverter_dataset='piqa'

model_names=('llama2')
sfl_datasets=("gsm8k")
seeds=(42 7 56)
for seed in "${seeds[@]}"; do
  for model_name in "${model_names[@]}"; do
    for sfl_dataset in "${sfl_datasets[@]}"; do
      case_name="SD${seed}-BiSR(b+f)@${model_name}@${sfl_dataset}"

      echo "Running evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
      python ../py/sim_with_attacker.py \
        --load_bits 8 --noise_mode "$noise_mode" \
        --case_name "$case_name" \
        --model_name "$model_name" \
        --split_points "$sps" \
        --global_round "$global_round" \
        --seed "$seed" \
        --dataset "$sfl_dataset" \
        --noise_scale "$noise_scale" \
        --exp_name "$exp_name" \
        --self_pt_enable "$self_pt_enable" \
        --client_num 1 \
        --data_shrink_frac "$data_shrink_frac" \
        --test_data_shrink_frac "$test_data_shrink_frac" \
        --evaluate_freq "$evaluate_freq" \
        --client_steps "$client_steps" \
        --lora_at_top "$lora_at_top" \
        --lora_at_trunk "$lora_at_trunk" \
        --lora_at_bottom "$lora_at_bottom" \
        --collect_all_layers "$collect_all_layers" \
        --dataset_label "$dataset_label" \
        --batch_size "$batch_size" \
        --tag_enable False \
        --gma_enable False \
        --gsma_enable False \
        --sma_enable False \
        --eia_enable False \
        --attacker_freq "$attacker_freq" \
        --attacker_samples "$attacker_samples" \
        --max_global_step "$max_global_step" \
        --sip_dataset "$sip_inverter_dataset" \
        --sip_prefix "nop" \
        --sip_b2tr_enable True \
        --sip_b2tr_layer 2 \
        --sip_model 'gru' \
        --sip_tr2t_enable False \
        --sip_training_epochs 10 \
        --sip_training_checkpoint_freq 1
    done
  done
done
