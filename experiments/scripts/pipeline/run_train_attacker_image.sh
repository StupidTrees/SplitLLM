seeds=(42)

datasets=('imagewoof')
attack_mode=('b2tr')
sp1s=(6)
sp2=999
model_name="vit-large"
save_checkpoint=True
log_to_wandb=False
dataset_train_frac=1.0
dataset_test_frac=1.0


for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for mode in "${attack_mode[@]}"; do
      for sp1 in "${sp1s[@]}"; do
        sps="${sp1}-${sp2}"
        echo "Running train_attacker_image.py with seed=$seed, dataset=$dataset,mode=$mode, split=$sps"
        python ../py/train_attacker_image.py \
          --model_name "$model_name" \
          --seed "$seed" \
          --dataset "$dataset" \
          --attack_model "vit" \
          --attack_mode "$mode" \
          --sps "$sps"\
          --save_checkpoint "$save_checkpoint" \
          --log_to_wandb "$log_to_wandb" \
          --dataset_train_frac "$dataset_train_frac" \
          --dataset_test_frac "$dataset_test_frac" \
          --epochs 20
      done
    done
  done
done
