# 实验: 要观察模型对于噪声的适应现象
seeds=(42)

dataset_label='train'
data_shrink_frac=0.08
test_data_shrink_frac=0.3

model_name='gpt2-large'

exp_name='[EXP]NoPeek_noise_adaption'

client_steps=2410
global_round=1
evaluate_freq=1200
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
attacker_freq=1200
attacker_samples=5
batch_size=6
sps='6-26'
# 噪声规模
sfl_noise_scales_dc=(30 70 90 110)
attacker_dc_scales=(0.0 8.0 10.0 16.0)

attacker_train_frac=0.2
attack_models=('gru')
atk_dataset='piqa'
sfl_dataset='piqa'
max_steps=2405
attack_layer=8
for seed in "${seeds[@]}"; do
  for attack_model in "${attack_models[@]}"; do
    for ns in "${sfl_noise_scales_dc[@]}"; do
      for atk_ns in "${attacker_dc_scales[@]}"; do
        # 先训练攻击模型

        file='train_inverter.py'
        if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
          file='train_inverter_moe.py'
        fi

        echo "Running $file with seed=$seed, dataset=$atk_dataset"
        python ../py/$file \
          --model_name "$model_name" \
          --seed "$seed" \
          --dataset "$atk_dataset" \
          --attack_model "$attack_model" \
          --attack_mode "tr2t" \
          --sps "6-${attack_layer}" \
          --save_checkpoint True \
          --log_to_wandb False \
          --noise_mode "dc" \
          --batch_size 6 --epochs 20 \
          --noise_scale_dc "$atk_ns" \
          --dataset_train_frac "$attacker_train_frac"

        attacker_prefix="dc:${atk_ns}"
        case_name="${sfl_dataset}[${ns}]<ATK${atk_dataset}[${atk_ns}]"

        # 将其用于攻击
        echo "Running evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
        python ../py/sim_with_attacker.py \
          --case_name "$case_name" \
          --model_name "$model_name" \
          --split_points "$sps" \
          --global_round "$global_round" \
          --seed "$seed" \
          --dataset "$sfl_dataset" \
          --noise_mode "dc" \
          --noise_scale_dc "$ns" \
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
          --collect_all_layers True \
          --dataset_label "$dataset_label" \
          --batch_size "$batch_size" \
          --tag_enable False \
          --gma_enable False \
          --gsma_enable False \
          --sma_enable False \
          --eia_enable False \
          --attacker_freq "$attacker_freq" \
          --attacker_samples "$attacker_samples" \
          --max_global_step "$max_steps" \
          --sip_dataset "$atk_dataset" \
          --sip_prefix "$attacker_prefix" \
          --sip_b2tr_enable True \
          --sip_b2tr_layer "$attack_layer" \
          --sip_b2tr_target_layer "$attack_layer" \
          --sip_model "$attack_model" \
          --sip_tr2t_enable False \
          --sip_train_frac "$attacker_train_frac" \
          --load_bits 8
      done
    done
  done
done
