# 实验：跨层DRA攻击

dataset_label='train'
exp_name='[CR]EIA_diff_noise_gaussian'
global_round=1
client_steps=500
attacker_prefix='normal'
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=900
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='llama2'

sps='6-26'
batch_size=2
attacker_freq=200
attacker_samples=5
max_global_step=605

noise_mode='gaussian'
noise_scale_gaussians=(6.5 7.0) # 6.0 5.0 4.0 3.0 2.0

attacker_datasets=("sensireplaced")
sfl_datasets=("piqa")
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")
eia_lr=0.11
eia_epc=72000
eia_temp=0.2
eia_wd=0.01
eia_depth=10
mapper_train_frac=1.0
seeds=(7 56)

for seed in "${seeds[@]}"; do
  for attacker_dataset in "${attacker_datasets[@]}"; do
    for sfl_dataset in "${sfl_datasets[@]}"; do
      for noise_scale_gaussian in "${noise_scale_gaussians[@]}"; do

        attacker_prefix="normal"
        if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
          attacker_prefix="gaussian"
        fi

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
          --dataset_test_frac 0.1 \
          --load_bits 8

        case_name="[${seed}]=${model_name}-${sfl_dataset}-${noise_mode}:${noise_scale_gaussian}"

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
          --noise_scale_gaussian "$noise_scale_gaussian" \
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
          --attacker_freq "$attacker_freq" \
          --attacker_samples "$attacker_samples" \
          --max_global_step "$max_global_step" \
          --sip_dataset "sensireplaced" \
          --sip_model "$attack_model" \
          --sip_prefix "$attacker_prefix" \
          --sip_b2tr_enable False \
          --sip_tr2t_enable False \
          --load_bits 8 \
          --eia_enable True \
          --eia_lr "$eia_lr" \
          --eia_epochs "$eia_epc" \
          --eia_temp "$eia_temp" \
          --eia_wd "$eia_wd" \
          --eia_mapped_to 1 \
          --eia_at "$eia_depth" \
          --eia_mapper_targets "${eia_depth}-1" \
          --eia_mapper_dataset "${attacker_dataset}" \
          --eia_mapper_train_frac "$mapper_train_frac"
      done
    done
  done

done
