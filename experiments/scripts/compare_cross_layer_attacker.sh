seeds=(42)

datasets=('wikitext')

model_name='gpt2-large'

exp_name='attacker_cross_different_sp'

sps=('2-31' '3-30' '4-29' '5-28' '6-27' '7-26' '8-25' '9-24' '10-23' '11-22' '12-21' '13-20' '14-19' '15-18' '16-17')
client_num=1
client_steps=250
global_round=1
noises=(0.0)
noise_mode="none"
client_from_scratch=('False')
attacker_b2tr_sp=15
attacker_tr2t_sp=15
attacker_prefixes=('normal')
attacker_freq=1
attacker_samples=99999999
attacker_search=('False')
dlg_enable="False"
dlg_adjust=(0)
dlg_beta=(0.8)
data_shrink_frac=0.5
test_data_shrink_frac=0.5
evaluate_freq=500
self_pt_enable='False'
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      for search in "${attacker_search[@]}"; do
        for ap in "${attacker_prefixes[@]}"; do
          for cfs in "${client_from_scratch[@]}"; do
            for mocker_a in "${dlg_adjust[@]}"; do
              for mocker_b in "${dlg_beta[@]}"; do
                for sp in "${sps[@]}"; do
                  echo "Running sfl_with_attacker.py with seed=$seed, dataset=$dataset, attacker_search=$search, noise=$noise"
                  python sfl_with_attacker.py \
                    --noise_mode "$noise_mode" \
                    --model_name "$model_name" \
                    --global_round "$global_round" \
                    --seed "$seed" \
                    --dataset "$dataset" \
                    --noise_scale "$noise" \
                    --exp_name "$exp_name" \
                    --split_points "$sp" \
                    --client_from_scratch "$cfs" \
                    --dlg_enable "$dlg_enable" \
                    --dlg_adjust "$mocker_a" \
                    --dlg_beta "$mocker_b" \
                    --attacker_b2tr_sp "$attacker_b2tr_sp" \
                    --attacker_tr2t_sp "$attacker_tr2t_sp" \
                    --attacker_prefix "$ap" \
                    --attacker_search "$search" \
                    --self_pt_enable "$self_pt_enable" \
                    --client_num "$client_num" \
                    --attacker_freq "$attacker_freq" \
                    --attacker_samples "$attacker_samples" \
                    --data_shrink_frac "$data_shrink_frac" \
                    --test_data_shrink_frac "$test_data_shrink_frac" \
                    --evaluate_freq "$evaluate_freq" \
                    --client_steps "$client_steps" \
                    --lora_at_top "$lora_at_top" \
                    --lora_at_trunk "$lora_at_trunk" \
                    --lora_at_bottom "$lora_at_bottom"
                done
              done
            done
          done
        done
      done
    done
  done
done
