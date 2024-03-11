seeds=(42)

datasets=('sanitized')
models=('gru')
attack_mode=('b2tr')
sp1s=(6 10 14)
sp2=999
model_name='llama2'
train_label='val'
test_label='test'
log_to_wandb=True
dataset_train_frac=1.0
dataset_test_frac=0.5

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for mode in "${attack_mode[@]}"; do
        for sp1 in "${sp1s[@]}"; do

          save_checkpoint=False

          if [ "$sp1" -eq 3 ] || [ "$sp1" -eq 4 ] || [ "$model_name" = 'llama2' ]; then
            save_checkpoint=true
          fi

          echo "Running train_attacker.py with seed=$seed, dataset=$dataset, model=$model, mode=$mode"
          python ../py/train_attacker.py \
            --exp_name "attacker-different-sp" \
            --model_name "$model_name" \
            --seed "$seed" \
            --dataset "$dataset" \
            --attack_model "$model" \
            --attack_mode "$mode" \
            --split_point_1 "$sp1" \
            --split_point_2 "$sp2" \
            --dataset_train_label "$train_label" \
            --dataset_test_label "$test_label" \
            --save_checkpoint "$save_checkpoint" \
            --log_to_wandb "$log_to_wandb" \
            --dataset_train_frac "$dataset_train_frac" \
            --dataset_test_frac "$dataset_test_frac" \
            --epochs 10
        done
      done
    done
  done
done
