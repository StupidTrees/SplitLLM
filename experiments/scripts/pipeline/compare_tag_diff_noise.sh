# 实验：不同方法的TAG
seeds=(42)

datasets=('piqa')
dataset_label='train'
test_data_label='test'
data_shrink_frac=0.5
test_data_shrink_frac=0.5

model_name='gpt2-large'

exp_name='[TEST]diff_noise'

client_num=1
client_steps=250
global_round=1
batch_size=6
dlg_enable=True
dlg_beta=(0.85)

evaluate_freq=400
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
attacker_b2tr_sp=6
attacker_tr2t_sp=6
attacker_freq=100
attacker_samples=5
sp='6-30'

# 噪声规模
noise_modes=('dc')
noise_beta_dcs=(0 0.2 0.6 0.8 1.0 1.5)
#noises_dxp=(0 50 20 15 10 5)
#noises_grad=(0 1.0 0.5 0.2)
# 是否初始化
init_with_dra=(True)
# 是否开启adjust
dlg_adjust=(0)
# dlg epoch
dlg_epochs=(18)
# dlg-DRA 正则参数
dlg_dra_regs=(0)
# dlg-temp 温度范围参数
dlg_temp_ranges=(0)
# Pre-FT 数据集
pre_ft_datasets=('')
pre_ft_data_label='train'
pre_ft_data_shrink_frac=0.1

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise_mode in "${noise_modes[@]}"; do
      for noise_beta_dc in "${noise_beta_dcs[@]}"; do
        #      for noise_dxp in "${noises_dxp[@]}"; do
        #        for noise_grad in "${noises_grad[@]}"; do
        # if noise_mode is none and (noise_grad>0 or noise_dxp>0.0) then continue, use bc to compare floats
        #          if [ "$noise_mode" == "none" ] && ([ $(echo "$noise_grad > 0" | bc -l) -eq 1 ] || [ $(echo "$noise_dxp > 0" | bc -l) -eq 1 ]); then
        #            continue
        #          fi
        #          # if noise_mode is dxp and (noise_dxp = 0 or noise_grad > 0) then skip the loop
        #          if [ "$noise_mode" == "dxp" ] && ([ $(echo "$noise_dxp == 0" | bc -l) -eq 1 ] || [ $(echo "$noise_grad > 0" | bc -l) -eq 1 ]); then
        #            continue
        #          fi
        #          # if noise_mode is grad and (noise_grad ==0 or noise_dxp>0) then continue, use bc
        #          if [ "$noise_mode" == "grad" ] && ([ $(echo "$noise_grad == 0" | bc -l) -eq 1 ] || [ $(echo "$noise_dxp > 0" | bc -l) -eq 1 ]); then
        #            continue
        #          fi
        #          # if noise_mode is both and (noise_grad ==0 or noise_dxp ==0) then continue
        #          if [ "$noise_mode" == "both" ] && ([ $(echo "$noise_grad == 0" | bc -l) -eq 1 ] || [ $(echo "$noise_dxp == 0" | bc -l) -eq 1 ]); then
        #            continue
        #          fi
        case_name="${noise_mode}${noise_beta_dc}"

        for mocker_b in "${dlg_beta[@]}"; do
          for dlg_epoch in "${dlg_epochs[@]}"; do
            for dlg_init_with_dra in "${init_with_dra[@]}"; do
              for dlg_dra_reg in "${dlg_dra_regs[@]}"; do
                for adjust in "${dlg_adjust[@]}"; do
                  for dlg_temp_range in "${dlg_temp_ranges[@]}"; do
                    for pre_ft_dataset in "${pre_ft_datasets[@]}"; do

                      echo "Running evaluate_tag_methods.py with seed=$seed, dataset=$dataset, noise=$noise_mode"
                      python ../py/sim_with_attacker.py \
                        --case_name "$case_name" \
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
                        --attacker_b2tr_sp "$attacker_b2tr_sp" \
                        --attacker_tr2t_sp "$attacker_tr2t_sp" \
                        --attacker_dataset "$dataset" \
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
                        --pre_ft_dataset "$pre_ft_dataset" \
                        --pre_ft_data_label "$pre_ft_data_label" \
                        --pre_ft_data_shrink_frac "$pre_ft_data_shrink_frac" \
                        --dlg_enable "$dlg_enable" \
                        --dlg_init_with_dra "$dlg_init_with_dra" \
                        --dlg_epochs "$dlg_epoch" \
                        --dlg_temp_range "$dlg_temp_range" \
                        --dlg_dra_reg "$dlg_dra_reg" \
                        --dlg_beta "$mocker_b" \
                        --dlg_adjust "$adjust" \
                        --batch_size "$batch_size" \
                        --noise_mode "$noise_mode" \
                        --noise_beta_dc "$noise_beta_dc"
                      #                          --noise_scale_dxp "$noise_dxp" \
                      #                          --noise_scale_grad "$noise_grad"
                      #                      done
                      #                    done
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
