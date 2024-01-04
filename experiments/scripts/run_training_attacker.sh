save_dir="/root/autodl-tmp/sfl/models"
model_cache_dir="/root/autodl-tmp/sfl/models"

seeds=(42)

datasets=('piqa' 'gsm8k')

models=('lstm' 'linear' 'trans-enc' 'trans-dec' 'gru')

attack_mode=('b2tr' 'tr2t')

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for mode in "${attack_mode[@]}"; do
        echo "Running train_attacker.py with seed=$seed, dataset=$dataset, model=$model, mode=$mode"
        python train_attacker.py --seed "$seed" --dataset "$dataset" --attack_model "$model" --save_dir "$save_dir" --model_cache_dir "$model_cache_dir" --attack_mode "$mode"
      done
    done
  done
done
