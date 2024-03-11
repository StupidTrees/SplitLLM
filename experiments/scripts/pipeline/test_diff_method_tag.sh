# 实验：用不同长度的尾部做TAG
seeds=(42)

datasets=('piqa')

model_name='flan-t5-base'

exp_name='tag_test'

client_num=1
client_steps=250
global_round=1
noise_mode="dxp"
attacker_b2tr_sp=2
attacker_tr2t_sp=2
dlg_enable=True
dlg_adjust=(0)
dlg_beta=(0.9)
data_shrink_frac=0.02
test_data_shrink_frac=0.5
evaluate_freq=500
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
attacker_freq=50
attacker_samples=5
sp='2-10'

# 噪声规模
noises=(0.1 0.2 0.5 0.8)
# 是否初始化
init_with_dra=(True)
# 是否开启adjust
adjust_mocker=(0 1 2)
# dlg epoch
dlg_epochs=(30 50 100)
# dlg-DRA 正则参数
dlg_dra_regs=(0)
# dlg-temp 温度范围参数
dlg_temp_ranges=(0)
dlg_further_ft=(0)
# Pre-FT 数据集
pre_ft_datasets=('')
pre_ft_data_label='train'
pre_ft_data_shrink_frac=0.1

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      for mocker_a in "${dlg_adjust[@]}"; do
        for mocker_b in "${dlg_beta[@]}"; do
          for dlg_epoch in "${dlg_epochs[@]}"; do
            for dlg_init_with_dra in "${init_with_dra[@]}"; do
              for dlg_dra_reg in "${dlg_dra_regs[@]}"; do
                for adjust in "${adjust_mocker[@]}"; do
                  for dlg_temp_range in "${dlg_temp_ranges[@]}"; do
                    for pre_ft_dataset in "${pre_ft_datasets[@]}"; do
                      for dlg_further_ft in "${dlg_further_ft[@]}"; do
                        echo "Running evaluate_tag_methods.py with seed=$seed, dataset=$dataset, noise=$noise"
                        python ../py/evaluate_tag_methods.py \
                          --noise_mode "$noise_mode" \
                          --model_name "$model_name" \
                          --global_round "$global_round" \
                          --seed "$seed" \
                          --dataset "$dataset" \
                          --noise_scale_dxp "$noise" \
                          --exp_name "$exp_name" \
                          --split_points "$sp" \
                          --dlg_enable "$dlg_enable" \
                          --dlg_adjust "$mocker_a" \
                          --dlg_beta "$mocker_b" \
                          --attacker_b2tr_sp "$attacker_b2tr_sp" \
                          --attacker_tr2t_sp "$attacker_tr2t_sp" \
                          --self_pt_enable "$self_pt_enable" \
                          --client_num "$client_num" \
                          --data_shrink_frac "$data_shrink_frac" \
                          --test_data_shrink_frac "$test_data_shrink_frac" \
                          --evaluate_freq "$evaluate_freq" \
                          --client_steps "$client_steps" \
                          --lora_at_top "$lora_at_top" \
                          --lora_at_trunk "$lora_at_trunk" \
                          --lora_at_bottom "$lora_at_bottom" \
                          --attacker_freq "$attacker_freq" \
                          --attacker_samples "$attacker_samples" \
                          --collect_all_layers False \
                          --dlg_init_with_dra "$dlg_init_with_dra" \
                          --dlg_adjust "$adjust" \
                          --data_shrink_frac "$data_shrink_frac" \
                          --dlg_epochs "$dlg_epoch" \
                          --dlg_dra_reg "$dlg_dra_reg" \
                          --dlg_temp_range "$dlg_temp_range" \
                          --pre_ft_dataset "$pre_ft_dataset" \
                          --pre_ft_data_label "$pre_ft_data_label" \
                          --pre_ft_data_shrink_frac "$pre_ft_data_shrink_frac" \
                          --dlg_further_ft "$dlg_further_ft"
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
