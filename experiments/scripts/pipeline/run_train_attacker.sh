seeds=(42)

datasets=('piqa')
models=('gru')
attack_mode=('b2tr')
sp1s=(6)
sp2=999
model_name="gpt2-large"
train_label='validation'
test_label='test'
save_checkpoint=True
log_to_wandb=False
dataset_train_frac=1.0
dataset_test_frac=0.3


for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for mode in "${attack_mode[@]}"; do
        for sp1 in "${sp1s[@]}"; do

          echo "Running train_attacker.py with seed=$seed, dataset=$dataset, model=$model, mode=$mode"
          python ../py/train_attacker.py \
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
            --epochs 50
        done
      done
    done
  done
done
