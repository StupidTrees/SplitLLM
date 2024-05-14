# 实验：不同方法的TAG
seeds=(42)

dataset_label='train'
test_data_label='test'
data_shrink_frac=0.06
test_data_shrink_frac=0.5

model_name='gpt2-large'

exp_name='[TEST]dra_diff_model'

client_num=1
client_steps=250
global_round=1
batch_size=2
dlg_enable=False

evaluate_freq=300
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
attacker_b2tr_sp=6
attacker_tr2t_sp=6
attacker_train_label='validation'
attacker_test_label='test'
attacker_freq=100
attacker_samples=5
sp='6-30'

# 噪声规模
noise_modes=("dxp")
noises_dxp=(0 30 20 15 12 10 8 6 4)
attack_models=('moe2')
datasets=('dialogsum')

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise_mode in "${noise_modes[@]}"; do
      for attack_model in "${attack_models[@]}"; do

        # 先训练攻击模型

        file='train_inverter.py'
        if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
          file='train_inverter_moe.py'
        fi

        echo "Running $file with seed=$seed, dataset=$dataset, model=$attack_model"
        python ../py/$file \
          --model_name "$model_name" \
          --seed "$seed" \
          --dataset "$dataset" \
          --attack_model "$attack_model" \
          --attack_mode 'b2tr' \
          --split_point_1 "$attacker_b2tr_sp" \
          --split_point_2 999 \
          --dataset_train_label "$attacker_train_label" \
          --dataset_test_label "$attacker_test_label" \
          --save_checkpoint True \
          --log_to_wandb False

        for noise_dxp in "${noises_dxp[@]}"; do
          echo "Running evaluate_tag_methods.py with seed=$seed, dataset=$dataset, noise=$noise_mode"
          python ../py/sim_with_attacker.py \
            --noise_mode "$noise_mode" \
            --model_name "$model_name" \
            --global_round "$global_round" \
            --seed "$seed" \
            --dataset "$dataset" \
            --dataset_label "$dataset_label" \
            --data_shrink_frac "$data_shrink_frac" \
            --test_data_label "$test_data_label" \
            --test_data_shrink_frac "$test_data_shrink_frac" \
            --exp_name "$exp_name" \
            --split_points "$sp" \
            --attacker_model "$attack_model" \
            --attacker_b2tr_sp "$attacker_b2tr_sp" \
            --attacker_tr2t_sp "$attacker_tr2t_sp" \
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
            --collect_all_layers False \
            --pre_ft_dataset "" \
            --dlg_enable "$dlg_enable" \
            --batch_size "$batch_size" \
            --attacker_train_label "$attacker_train_label" \
            --noise_mode "$noise_mode" \
            --noise_scale_dxp "$noise_dxp"
        done
      done
    done
  done
done
