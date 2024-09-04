# 实验：跨层DRA攻击

dataset_label='train'
exp_name='[CR]BiSR_diff_noise_nopeek'
global_round=1
client_steps=2440
attacker_prefix='normal'
data_shrink_frac=0.08
test_data_shrink_frac=0.3

self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='gpt2-large'

sps='6-26'
attacker_sp=8
batch_size=6
attacker_freq=1200
attacker_samples=5
max_global_step=2405

noise_mode='dc'
noise_scale_dcs=(10.0 30.0 50.0 70.0 90.0 110.0 130.0) #12.0 6.0 5.0 4.0
attack_models=('moe')

attacker_datasets=("sensireplaced")
sfl_datasets=("piqa")
seeds=(42 66 89)
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")
gma_lr=0.09
gma_beta=0.85
gma_epc=18
gma_init_temp=1.2
gsma_lr=0.005
gsma_epc=64
gsma_wd=0.02
sma_lr=0.005
sma_epc=256
sma_wd=0.02

for seed in "${seeds[@]}"; do
  for attacker_dataset in "${attacker_datasets[@]}"; do
    for sfl_dataset in "${sfl_datasets[@]}"; do
      for noise_scale_dc in "${noise_scale_dcs[@]}"; do
        for attack_model in "${attack_models[@]}"; do

          evaluate_freq=9000
          if [ "$attack_model" = "moe" ]; then
            evaluate_freq=2400
          fi

          file='train_inverter.py'
          if [ "$attack_model" = "moe" ]; then
            file='train_inverter_moe.py'
          fi

          train_mode="dc"
          if [ "$attack_model" = "gru" ]; then
            train_mode="none"
          fi

          echo "Running $file with seed=$seed, dataset=$attacker_dataset"
          python ../py/$file \
            --model_name "$model_name" \
            --seed "$seed" \
            --dataset "$attacker_dataset" \
            --attack_model "$attack_model" \
            --attack_mode "tr2t" \
            --sps "6-${attacker_sp}" \
            --save_checkpoint True \
            --log_to_wandb False \
            --noise_mode "$train_mode" \
            --epochs 15 \
            --epochs_gating 10 \
            --epochs_ft 4 \
            --noise_scale_dc "$noise_scale_dc" \
            --dataset_train_frac 1.0 \
            --dataset_test_frac 0.1
          #            --require_prefix "dc-bk"

          attacker_prefix="normal"
          if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
            attacker_prefix="dc"
          fi

          case_name="[${seed}]=${model_name}-${sfl_dataset}-${noise_mode}:${noise_scale_dc}<${attack_model}"

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
            --noise_scale_dc "$noise_scale_dc" \
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
            --gma_enable True \
            --gsma_enable True --sma_enable True \
            --eia_enable False \
            --attacker_freq "$attacker_freq" \
            --attacker_samples "$attacker_samples" \
            --max_global_step "$max_global_step" \
            --sip_dataset "sensireplaced" \
            --sip_model "$attack_model" \
            --sip_prefix "$attacker_prefix" \
            --sip_b2tr_enable True \
            --sip_b2tr_layer "$attacker_sp" \
            --sip_b2tr_target_layer "$attacker_sp" \
            --sip_tr2t_enable False \
            --gma_lr "$gma_lr" \
            --gma_beta "$gma_beta" \
            --gma_epochs "$gma_epc" \
            --gma_init_temp "$gma_init_temp" \
            --gsma_lr "$gsma_lr" \
            --gsma_epochs "$gsma_epc" \
            --gsma_wd "$gsma_wd" \
            --gsma_at "11" \
            --load_bits 8 \
            --smalr "$sma_lr" \
            --sma_epochs "$sma_epc" \
            --sma_wd "$sma_wd" \
            --sma_at "11"
        done
      done
    done
  done
done
