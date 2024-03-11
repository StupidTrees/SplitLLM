seeds=(42)

datasets=('dialogsum,wikitext,gsm8k')
models=('moe2')
attack_mode=('b2tr')
sp1s=(6)
sp2=999
model_name="gpt2-large"
save_checkpoint=True
log_to_wandb=False
dataset_train_frac=1.0
dataset_test_frac=0.3
load_bits=8
epochs_expert=20
epochs_gating=10
noise_mode='dxp'

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for mode in "${attack_mode[@]}"; do
        for sp1 in "${sp1s[@]}"; do
          echo "Running train_attacker_moe.py with seed=$seed, dataset=$dataset, model=$model, mode=$mode"
          python ../py/train_attacker_moe.py \
            --model_name "$model_name" \
            --seed "$seed" \
            --dataset "$dataset" \
            --attack_model "$model" \
            --attack_mode "$mode" \
            --split_point_1 "$sp1" \
            --split_point_2 "$sp2" \
            --save_checkpoint "$save_checkpoint" \
            --log_to_wandb "$log_to_wandb" \
            --dataset_train_frac "$dataset_train_frac" \
            --dataset_test_frac "$dataset_test_frac" \
            --load_bits "$load_bits" \
            --epochs_expert "$epochs_expert" \
            --epochs_gating "$epochs_gating" \
            --noise_mode "$noise_mode"\
            --skip_exist False
        done
      done
    done
  done
done
