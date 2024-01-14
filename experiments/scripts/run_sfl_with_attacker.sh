seeds=(26)

datasets=('piqa')

model_name='gpt2'
attacker_search=('False' 'True')
exp_name='sfl_with_attacker_scratch'

sp1=2
sp2=10
global_round=100
attacker_prefixes=('normal')
noises=(0.0)
noise_mode="none"
client_from_scratch=('True' 'False')

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      for search in "${attacker_search[@]}"; do
        for ap in "${attacker_prefixes[@]}"; do
          for cfs in "${client_from_scratch[@]}"; do
            echo "Running sfl_with_attacker.py with seed=$seed, dataset=$dataset, attacker_search=$search, noise=$noise"
            python sfl_with_attacker.py \
              --noise_mode "$noise_mode" \
              --attacker_prefix "$ap" \
              --model_name "$model_name" \
              --global_round "$global_round" \
              --seed "$seed" \
              --dataset "$dataset" \
              --attacker_search $search \
              --noise_scale "$noise" \
              --exp_name "$exp_name" \
              --split_point_1 "$sp1" \
              --split_point_2 "$sp2" \
              --client_from_scratch "$cfs"
          done
        done
      done
    done
  done
done
