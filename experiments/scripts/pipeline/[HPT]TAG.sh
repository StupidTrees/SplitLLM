# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[HPT]TAG'
global_round=1
client_steps=500
noise_scale=0.0
noise_mode="none"
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=300
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='falcon'

sps="6-27"
batch_size=2

attacker_freq=200
attacker_samples=1
max_global_step=405

sfl_datasets=("piqa")


tag_lrs=(0.09 0.06 0.11)
tag_betas=(0.6 0.85)
tag_epochs=(400 600 800 1000 1200)

#chatglm
#tag_lrs=(0.09 0.06 0.11)
#tag_betas=(0.6 0.85)
#tag_epochs=(400 600 800 1000 1200)

# gpt2
#tag_lrs=(0.09 0.06 0.11)
#tag_betas=(0.6 0.85)
#tag_epochs=(400 600 800 1000 1200)

## llama2
#tag_lrs=(0.09 0.06 0.11)
#tag_betas=(0.6 0.85)
#tag_epochs=(400 600 800 1000)
# 0.05 0.001 0.1)


for sfl_dataset in "${sfl_datasets[@]}"; do
  for tag_lr in "${tag_lrs[@]}"; do
    for tag_epc in "${tag_epochs[@]}"; do
      for tag_beta in "${tag_betas[@]}"; do

        case_name="TAG@${model_name}@${sfl_dataset}@${tag_lr}@${tag_beta}@${tag_epc}"

        # 将其用于攻击
        echo "Running evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
        python ../py/sim_with_attacker.py \
          --noise_mode "$noise_mode" \
          --case_name "$case_name" \
          --model_name "$model_name" \
          --split_points "$sps" \
          --global_round "$global_round" \
          --seed "$seed" \
          --dataset "$sfl_dataset" \
          --noise_scale "$noise_scale" \
          --exp_name "$exp_name" \
          --sip_b2tr_enable False \
          --sip_tr2t_enable False \
          --self_pt_enable "$self_pt_enable" \
          --client_num "1" \
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
          --tag_enable True \
          --gma_enable False \
          --gsma_enable False \
          --sma_enable False \
          --eia_enable False --attacker_freq "$attacker_freq" \
          --attacker_samples "$attacker_samples" \
          --max_global_step "$max_global_step" \
          --tag_beta "$tag_beta" \
          --tag_lr "$tag_lr" \
          --tag_epochs "$tag_epc"\
          --load_bits 32
      done
    done
  done
done
