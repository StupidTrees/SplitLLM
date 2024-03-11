# 实验：用不同长度的尾部做TAG
seeds=(42)

datasets=('wikitext')

model_name='gpt2-large'

exp_name='tag_across_layer'

client_num=1
client_steps=250
global_round=1
noises=(0.0)
noise_mode="none"
attacker_b2tr_sp=15
attacker_tr2t_sp=15
dlg_enable=True
dlg_adjust=(0)
dlg_beta=(0.8)
data_shrink_frac=0.5
test_data_shrink_frac=0.5
evaluate_freq=500
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
attacker_freq=300
attacker_samples=20

# 不同的尾部切分
sps=('6-21' '6-23')

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      for mocker_a in "${dlg_adjust[@]}"; do
        for mocker_b in "${dlg_beta[@]}"; do
          for sp in "${sps[@]}"; do
            echo "Running evaluate_tag_diff_layer.py with seed=$seed, dataset=$dataset, noise=$noise"
            python ../py/evaluate_tag_diff_layer.py \
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
              --collect_all_layers False
          done
        done
      done
    done
  done
done
