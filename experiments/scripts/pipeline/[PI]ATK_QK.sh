model_names=('bert-large' 'gpt2-large') #'bert-base' 'bert-large' 'gpt2' 'gpt2-large') #
seeds=(42 7 56)                         #  7 56)
targets=('o5')
modes=('none' 'perm' 'random')

for seed in "${seeds[@]}"; do
  for model_name in "${model_names[@]}"; do

    if [ "$model_name" == "bert-base" ] || [ "$model_name" == "bert-large" ]; then
      datasets=("qnli" "cola" "mrpc" 'stsb')
    fi
    if [ "$model_name" == "gpt2" ] || [ "$model_name" == "gpt2-large" ]; then
      datasets=("wikitext103" "wikitext")
    fi

    for dataset in "${datasets[@]}"; do

      for target in "${targets[@]}"; do

        for mode in "${modes[@]}"; do

          python ../py/train_inverter_qk.py \
            --model_name "$model_name" \
            --lr 1e-3 \
            --dataset "$dataset" \
            --log_to_wandb True \
            --seed "$seed" \
            --epochs 5 \
            --target "$target" \
            --batch_size 6 \
            --sps "1-12" \
            --checkpoint_freq 1 \
            --train_dataset "sensireplaced" \
            --save_checkpoint True \
            --uni_length 512 \
            --mode "$mode" \
            --debug False
        done

      done

    done

  done

done
