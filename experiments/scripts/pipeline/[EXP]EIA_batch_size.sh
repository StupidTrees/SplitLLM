# 实验：Embedding Inversion Attack在不同数据集和模型上的实验

dataset_label='train'
exp_name='[EXP]EIA_batch_size'
global_round=1
client_steps=500
noise_scale=0.0
noise_mode="none"
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=800
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=False

eia_depth=6
sps="$eia_depth-27"

attacker_freq=300
attacker_samples=5
max_global_step=605
mapper_train_frac=1.0

attacker_dataset='sensireplaced'
seeds=(42)
model_names=('gpt2-large')
load_bits=8
sfl_datasets=("piqa-mini")

batch_sizes=(1 2 3 4 5 6 7 8)

for batch_size in "${batch_sizes[@]}"; do

  for seed in "${seeds[@]}"; do
    for model_name in "${model_names[@]}"; do
      for sfl_dataset in "${sfl_datasets[@]}"; do
        # 先训练Mapper

        echo "Running train_mapper.py with seed=$seed, dataset=$attacker_dataset"
        python ../py/train_mapper.py \
          --model_name "$model_name" \
          --seed "$seed" \
          --dataset "$attacker_dataset" \
          --attack_mode "b2tr" \
          --target "${eia_depth}-1" \
          --save_checkpoint True \
          --log_to_wandb False \
          --epochs 10 \
          --dataset_train_frac "$mapper_train_frac" \
          --dataset_test_frac 0.1 --load_bits "$load_bits"

        if [ "$model_name" == "llama2" ]; then
          eia_lr=0.11
          eia_epc=72000
          eia_temp=0.2
          eia_wd=0.01
        fi

        if [ "$model_name" == "gpt2-large" ]; then
          eia_lr=0.11
          eia_epc=24000
          eia_temp=0.3
          eia_wd=0.01
        fi

        if [ "$model_name" == "chatglm" ]; then
          eia_lr=0.11
          eia_epc=24000
          eia_temp=0.3
          eia_wd=0.01
        fi

        case_name="${model_name}-${sfl_dataset}-bs=${batch_size}"

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
          --eia_enable True \
          --eia_lr "$eia_lr" \
          --eia_epochs "$eia_epc" \
          --eia_temp "$eia_temp" \
          --eia_wd "$eia_wd" \
          --eia_mapped_to 1 \
          --eia_mapper_targets "${eia_depth}-1" \
          --eia_mapper_dataset "${attacker_dataset}" \
          --eia_mapper_train_frac "$mapper_train_frac" \
          --load_bits "$load_bits"
      done
    done
  done
done
