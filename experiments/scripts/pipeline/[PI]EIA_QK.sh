model_names=('bert-large') #('bert-base' 'bert-large') #('gpt2' 'gpt2-large') #'bert-base' 'bert-large'

sample_num=20
seeds=(42 7 56)
batch_size=4
targets=('o5' 'o6')  #'hidden_pi' 'random')
methods=('bre') # 'bre'
modes=('none')
wd=0.01

for seed in "${seeds[@]}"; do
  for mode in "${modes[@]}"; do
    for method in "${methods[@]}"; do
      for model_name in "${model_names[@]}"; do
        for target in "${targets[@]}"; do
          if [ "$model_name" == "bert-base" ] || [ "$model_name" == "bert-large" ]; then

            datasets=("mrpc") # qnli" "cola" "mrpc" "rte
            if [ "$method" == "eia" ]; then
              lr=0.1
              epochs=2400
              if [ "$target" == "o6" ]; then
                lr=0.15
                wd=0.18
                epochs=6000
              fi
            fi
            if [ "$method" == "bre" ]; then
              lr=0.1
              epochs=9000
              if [ "$target" == "o6" ]; then
                lr=0.15
                wd=0.18
                epochs=6000
              fi
            fi

          fi

          if [ "$model_name" == "gpt2" ] || [ "$model_name" == "gpt2-large" ]; then
            datasets=("wikitext103" "wikitext")
            lr=0.1
            wd=0.01
            if [ "$method" == "eia" ]; then
              epochs=2800
              if [ "$target" == "qk" ]; then
                lr=0.2
                wd=0.1
              fi
            fi
            if [ "$method" == "bre" ]; then
              lr=0.1
              epochs=6000
              if [ "$target" == "o6" ]; then
                lr=0.11
                wd=0.1
                epochs=3000
              fi
            fi
          fi

          if [ "$mode" == "random" ] || [ "$mode" == "perm" ]; then
            epochs=600
          fi

          for dataset in "${datasets[@]}"; do

            if [ "$dataset" == "mrpc" ] && [ "$model_name" == "bert-large" ] && [ "$target" != "qk" ]; then
              echo "Skipping $dataset for $model_name with target $target"
              lr=0.03
              wd=0.05
              epochs=10000
            fi

            echo "Running eia_attack_qk.py with model_name=$model_name, dataset=$dataset"
            python ../py/eia_attack_qk.py \
              --sps "1-12" \
              --model_name "$model_name" \
              --lr "$lr" \
              --dataset "$dataset" \
              --epochs "$epochs" \
              --seed "$seed" \
              --target "$target" \
              --sample_num "$sample_num" \
              --log_to_wandb True --batch_size "$batch_size" \
              --method "$method" \
              --mode "$mode" \
              --uni_length 512 \
              --wd "$wd"
          done
        done
      done
    done
  done
done
