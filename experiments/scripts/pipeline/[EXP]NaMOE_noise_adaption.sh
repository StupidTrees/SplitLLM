# 实验: 要观察模型对于噪声的适应现象
seeds=(42)

dataset_label='train'
test_data_label='test'
data_shrink_frac=0.08
test_data_shrink_frac=0.3

model_name='llama2'

exp_name='[EXP]NaMOE_noise_adaption'

client_num=1
client_steps=600
global_round=1
dlg_enable=False

evaluate_freq=300
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
attacker_freq=200
attacker_samples=20
batch_size=1
sps='6-26'
attacker_sp=6

# 噪声规模
noise_modes=("gaussian")
sfl_noise_scales_gaussian=(2.0 3.0 5.0)
sfl_noise_scales_dxp=(0.0 0.3 0.5 0.7)
attacker_noises_gaussian=(0.0)
attacker_noises_dxp=(0.0 0.2 0.4 0.6 0.8)
attacker_train_frac=0.2
attack_models=('gru')
datasets=('piqa')
sfl_dataset='piqa'
max_steps=1200

for seed in "${seeds[@]}"; do
  for atk_dataset in "${datasets[@]}"; do
    for noise_mode in "${noise_modes[@]}"; do
      for attack_model in "${attack_models[@]}"; do
        noise_scale_name='noise_scale_dxp'
        sfl_noise_scales=("${sfl_noise_scales_dxp[@]}")
        attacker_noise_scales=("${attacker_noises_dxp[@]}")
        if [ "$noise_mode" = "gaussian" ]; then
          noise_scale_name='noise_scale_gaussian'
          sfl_noise_scales=("${sfl_noise_scales_gaussian[@]}")
          attacker_noise_scales=("${attacker_noises_gaussian[@]}")
        fi

        for ns in "${sfl_noise_scales[@]}"; do
          for atk_ns in "${attacker_noise_scales[@]}"; do
            # 先训练攻击模型

            file='train_attacker.py'
            if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
              file='train_attacker_moe.py'
            fi

            attack_mode='b2tr'
            if [ "$noise_mode" = "gaussian" ]; then
              attack_mode='tr2t'
              attacker_sp=26
            fi

            echo "Running $file with seed=$seed, dataset=$atk_dataset, mode=$attack_mode"
            python ../py/$file \
              --model_name "$model_name" \
              --seed "$seed" \
              --dataset "$atk_dataset" \
              --attack_model "$attack_model" \
              --attack_mode "$attack_mode" \
              --sps "$sps" \
              --save_checkpoint True \
              --log_to_wandb False \
              --noise_mode "$noise_mode" \
              --epochs 20 \
              --"${noise_scale_name}" "$atk_ns" \
              --dataset_train_frac "$attacker_train_frac"

            attacker_prefix="${noise_mode}:${atk_ns}"
            if [ "$noise_mode" = "none" ]; then
              attacker_prefix="normal"
            fi
            case_name="${sfl_dataset}[${noise_mode}-${ns}]<ATK${atk_dataset}[${noise_mode}-${atk_ns}]"

            echo "Running ${case_name}"
            python ../py/evaluate_tag_methods.py \
              --noise_mode "$noise_mode" \
              --exp_name "$exp_name" \
              --model_name "$model_name" \
              --global_round "$global_round" \
              --seed "$seed" \
              --dataset "$sfl_dataset" \
              --dataset_label "$dataset_label" \
              --data_shrink_frac "$data_shrink_frac" \
              --test_data_label "$test_data_label" \
              --test_data_shrink_frac "$test_data_shrink_frac" \
              --case_name "$case_name" \
              --split_points "$sps" \
              --attacker_model "$attack_model" \
              --attacker_dataset "$atk_dataset" \
              --attacker_b2tr_sp "$attacker_sp" \
              --attacker_tr2t_sp "$attacker_sp" \
              --self_pt_enable "$self_pt_enable" \
              --client_num "$client_num" \
              --data_shrink_frac "$data_shrink_frac" \
              --evaluate_freq "$evaluate_freq" \
              --client_steps "$client_steps" \
              --lora_at_top "$lora_at_top" \
              --lora_at_trunk "$lora_at_trunk" \
              --lora_at_bottom "$lora_at_bottom" \
              --attacker_freq "$attacker_freq" \
              --attacker_samples "$attacker_samples" \
              --attacker_prefix "$attacker_prefix" \
              --attacker_train_frac "$attacker_train_frac" \
              --collect_all_layers True \
              --pre_ft_dataset "" \
              --dlg_enable "$dlg_enable" \
              --batch_size "$batch_size" \
              --"${noise_scale_name}" "${ns}" \
              --max_global_step "$max_steps"
          done
        done
      done
    done
  done
done
