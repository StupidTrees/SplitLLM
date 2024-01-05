
seeds=(42)

datasets=('piqa' 'gsm8k')

models=('gru')

# use a set of float numbers in shell
noises=(0.0 0.4 1.5 1.5 2.0)

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for noise in "${noises[@]}"; do
        echo "Running sfl_with_attacker.py with seed=$seed, dataset=$dataset, model=$model, noise=$noise"
        python sfl_with_attacker.py --seed "$seed" --dataset "$dataset" --attack_model "$model" --noise_scale "$noise"
      done
    done
  done
done
