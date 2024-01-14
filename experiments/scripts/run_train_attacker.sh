seeds=(42)

datasets=('piqa')
models=('gru')
attack_mode=('b2tr' 'tr2t')
sp1=2
sp2=10
model_name='gpt2'

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for mode in "${attack_mode[@]}"; do
        echo "Running train_attacker.py with seed=$seed, dataset=$dataset, model=$model, mode=$mode"
        python train_attacker.py --model_name "$model_name" --seed "$seed" --dataset "$dataset" --attack_model "$model" --attack_mode "$mode" --split_point_1 "$sp1" --split_point_2 "$sp2"
      done
    done
  done
done
