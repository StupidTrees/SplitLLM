# 实验：用不同长度的尾部做TAG
seeds=(42)

model_name='gpt2-large'

exp_name='inference_stage_attack'

noise_mode="dxp"
attacker_b2tr_sp=8
attacker_tr2t_sp=23
dlg_enable=True
dlg_adjust=(0)
dlg_beta=(0.8)
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
sp='8-23'

# 噪声规模
noises=(12 13 14 15 16 17 18 19 20)

datasets=('wikitext')
dataset_label='test'
batch_size=2
data_shrink_frac=0.33
pre_ft_datasets=('' 'wikitext')
pre_ft_data_label='train'
pre_ft_data_shrink_frac=0.08

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      for mocker_a in "${dlg_adjust[@]}"; do
        for mocker_b in "${dlg_beta[@]}"; do
          for pre_ft_dataset in "${pre_ft_datasets[@]}"; do
            echo "Running evaluate_inference.py with seed=$seed, dataset=$dataset, noise=$noise"
            python ../py/evaluate_inference.py \
              --noise_mode "$noise_mode" \
              --model_name "$model_name" \
              --batch_size "$batch_size" \
              --seed "$seed" \
              --dataset "$dataset" \
              --dataset_label "$dataset_label" \
              --noise_scale "$noise" \
              --exp_name "$exp_name" \
              --split_points "$sp" \
              --dlg_enable "$dlg_enable" \
              --dlg_adjust "$mocker_a" \
              --dlg_beta "$mocker_b" \
              --attacker_b2tr_sp "$attacker_b2tr_sp" \
              --attacker_tr2t_sp "$attacker_tr2t_sp" \
              --self_pt_enable "$self_pt_enable" \
              --data_shrink_frac "$data_shrink_frac" \
              --lora_at_top "$lora_at_top" \
              --lora_at_trunk "$lora_at_trunk" \
              --lora_at_bottom "$lora_at_bottom" \
              --collect_all_layers False \
              --data_shrink_frac "$data_shrink_frac" \
              --pre_ft_dataset "$pre_ft_dataset" \
              --pre_ft_data_label "$pre_ft_data_label" \
              --pre_ft_data_shrink_frac "$pre_ft_data_shrink_frac"
          done
        done
      done
    done
  done
done
