# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[HPT]EIA'
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

model_name='chatglm'

eia_depth=6
sps="$eia_depth-27"
batch_size=2

attacker_freq=200
attacker_samples=1
max_global_step=405
mapper_train_frac=1.0
mapper_datasets=("sensireplaced")
sfl_datasets=("piqa")

eia_enable=True
eia_mapped_to=1

# Falcon
#eia_lrs=(0.11 0.09)
#eia_epochs=(12000 24000 36000)
#eia_temps=(0.5 0.3)
#eia_wds=(0.01)

#GPT2
#eia_lrs=(0.11 0.09 0.06)
#eia_epochs=(12000 24000)
#eia_temps=(0.5 0.3 0.2)
#eia_wds=(0.01)

# chatglm
eia_lrs=(0.09)
eia_epochs=(48000)
eia_temps=(0.3)
eia_wds=(0.01)
load_bits=8
# LLaMA2
#eia_lrs=(0.001 0.06 0.11)
#eia_epochs=(72000)
#eia_temps=(0.5 0.3 0.2)
#eia_wds=(0.01)

config_file='/data/stupidtree/project/SFL-LLM/experiments/scripts/config/mapper.yaml'

for mapper_dataset in "${mapper_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    for eia_lr in "${eia_lrs[@]}"; do
      for eia_epc in "${eia_epochs[@]}"; do
        for eia_temp in "${eia_temps[@]}"; do
          for eia_wd in "${eia_wds[@]}"; do

            # 先训练Mapper
            echo "Running train_mapper.py with seed=$seed, dataset=$mapper_dataset"
            python ../py/train_mapper.py \
             --config_file "$config_file" \
              --model_name "$model_name" \
              --seed "$seed" \
              --dataset "$mapper_dataset" \
              --attack_mode "b2tr" \
              --target "${eia_depth}-1" \
              --save_checkpoint True \
              --log_to_wandb False \
              --epochs 20 \
              --dataset_train_frac "$mapper_train_frac" \
              --dataset_test_frac 0.1\
              --lr 0.0005\
              --wd 0.01\
              --load_bits "$load_bits"

            case_name="EIA@${model_name}${eia_depth}_lr=${eia_lr},epc=${eia_epc},temp=${eia_temp},wd=${eia_wd}"

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
              --noise_scale_dxp "$noise_scale" \
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
              --tag_enable False \
              --gma_enable False \
              --gsma_enable False \
              --sma_enable False \
              --attacker_freq "$attacker_freq" \
              --attacker_samples "$attacker_samples" \
              --max_global_step "$max_global_step" \
              --eia_enable "$eia_enable" \
              --eia_lr "$eia_lr" \
              --eia_epochs "$eia_epc" \
              --eia_temp "$eia_temp" \
              --eia_wd "$eia_wd" \
              --eia_mapped_to "$eia_mapped_to" \
              --mapper_target "${eia_depth}-1" \
              --mapper_dataset "${mapper_dataset}" \
              --mapper_train_frac "$mapper_train_frac"\
              --load_bits "$load_bits"
          done
        done
      done
    done
  done
done
