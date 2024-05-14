seeds=(42)

datasets=('wikitext')
models=('gru')
attack_mode=('b2tr')
sp1s=(6)
sp2=999
model_name="gpt2-large"
save_checkpoint=True
log_to_wandb=False
dataset_train_frac=1.0
dataset_test_frac=1.0
load_bits=8
noise_mode='dxp'
noise_scale_gaussian=0.2
noise_scale_dxp=0.2

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for mode in "${attack_mode[@]}"; do
        for sp1 in "${sp1s[@]}"; do

          echo "Running train_attacker_no_pretrained.py with seed=$seed, dataset=$dataset, model=$model, mode=$mode"
          python ../py/train_inverter_no_pretrained.py \
            --model_name "$model_name" \
            --seed "$seed" \
            --dataset "$dataset" \
            --attack_model "$model" \
            --attack_mode "$mode" \
            --sps "$sp1-$sp2" \
            --save_checkpoint "$save_checkpoint" \
            --log_to_wandb "$log_to_wandb" \
            --dataset_train_frac "$dataset_train_frac" \
            --dataset_test_frac "$dataset_test_frac" \
            --noise_mode "$noise_mode"\
            --noise_scale_gaussian "$noise_scale_gaussian"\
            --noise_scale_dxp "$noise_scale_dxp"\
            --load_bits "$load_bits"\
            --epochs 20\
            --checkpoint_freq 5
        done
      done
    done
  done
done
