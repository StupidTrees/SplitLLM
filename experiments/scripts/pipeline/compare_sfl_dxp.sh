seeds=(42)

datasets=('wikitext')

model_name='gpt2-large'

exp_name='compare_sfl_dxp'

sps='8-30'
global_round=1
noises=(5.0 10.0 20.0)
noise_mode="dxp"
client_from_scratch=('False')
attacker_prefixes=('normal')
attacker_search=(True)
attacker_freq=100
attacker_samples=50
dlg_enable=False
dlg_adjust=(0)
dlg_beta=(0.8)
self_pt_enable=False

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      for search in "${attacker_search[@]}"; do
        for ap in "${attacker_prefixes[@]}"; do
          for cfs in "${client_from_scratch[@]}"; do
            for mocker_a in "${dlg_adjust[@]}"; do
              for mocker_b in "${dlg_beta[@]}"; do
                echo "Running sfl_with_attacker.py with seed=$seed, dataset=$dataset, attacker_search=$search, noise=$noise"
                python ../py/sfl_with_attacker.py \
                  --noise_mode "$noise_mode" \
                  --model_name "$model_name" \
                  --global_round "$global_round" \
                  --seed "$seed" \
                  --dataset "$dataset" \
                  --noise_scale_dxp "$noise" \
                  --exp_name "$exp_name" \
                  --split_points "$sps" \
                  --client_from_scratch "$cfs" \
                  --dlg_enable "$dlg_enable" \
                  --dlg_adjust "$mocker_a" \
                  --dlg_beta "$mocker_b" \
                  --attacker_prefix "$ap" \
                  --attacker_search "$search" \
                  --self_pt_enable "$self_pt_enable" \
                  --attacker_samples "$attacker_samples" \
                  --attacker_freq "$attacker_freq"
              done
            done
          done
        done
      done
    done
  done
done
