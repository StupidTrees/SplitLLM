seeds=(42)

datasets=('wikitext')
models=('gru')
attack_mode=('b2tr')
sp1s=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
sp2=999
model_name='gpt2-large'
train_label='validation'
test_label='test'
save_checkpoint='False'
log_to_wandb='False'

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for mode in "${attack_mode[@]}"; do
        for sp1 in "${sp1s[@]}"; do
          echo "Running train_attacker.py with seed=$seed, dataset=$dataset, model=$model, mode=$mode"
          python train_attacker.py \
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
            --log_to_wandb "$log_to_wandb"
        done
      done
    done
  done
done
