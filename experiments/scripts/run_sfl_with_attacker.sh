seeds=(54)

datasets=('piqa')

model_name='gpt2'

exp_name='sfl_tag_hypersearch'

sp1=2
sp2=10
global_round=50
attacker_prefixes=('normal')
noises=(1.0 3.0 5.0)
noise_mode="dxp"
client_from_scratch=('False')
attacker_enable="False"
attacker_search=('False')
mocker_enable="True"
mocker_adjust=(0 10 50)
mocker_beta=(0.6 0.3 0.1)

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      for search in "${attacker_search[@]}"; do
        for ap in "${attacker_prefixes[@]}"; do
          for cfs in "${client_from_scratch[@]}"; do
            for mocker_a in "${mocker_adjust[@]}"; do
              for mocker_b in "${mocker_beta[@]}"; do
                echo "Running sfl_with_attacker.py with seed=$seed, dataset=$dataset, attacker_search=$search, noise=$noise"
                python sfl_with_attacker.py \
                  --noise_mode "$noise_mode" \
                  --model_name "$model_name" \
                  --global_round "$global_round" \
                  --seed "$seed" \
                  --dataset "$dataset" \
                  --noise_scale "$noise" \
                  --exp_name "$exp_name" \
                  --split_point_1 "$sp1" \
                  --split_point_2 "$sp2" \
                  --client_from_scratch "$cfs" \
                  --mocker_enable "$mocker_enable" \
                  --mocker_adjust "$mocker_a" \
                  --mocker_beta "$mocker_b"\
                  --attacker_enable "$attacker_enable" \
                  --attacker_prefix "$ap" \
                  --attacker_search "$search"
              done
            done
          done
        done
      done
    done
  done
done
