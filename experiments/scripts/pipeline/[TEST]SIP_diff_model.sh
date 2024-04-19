# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[TEST]SIP_diff_model'
client_num=1
global_round=1
client_steps=500
noise_scale=0.0
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=300
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='llama2'

sps='6-27'
attacker_sp=6
batch_size=2
attacker_freq=200
attacker_samples=10
max_global_step=610
atk_train_frac=0.3
noise_mode='dxp'
noise_scale_dxps=(0.1)
attacker_noise_scale_dxp=0.2
attack_models=('gruattn')

dlg_enable=False
dlg_raw_enable=False

attacker_datasets=("piqa")
sfl_datasets=("wikitext")
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    for noise_scale_dxp in "${noise_scale_dxps[@]}"; do
      for attack_model in "${attack_models[@]}"; do
        noise_scale="$noise_scale_dxp"
        # 先训练攻击模型

        file='train_attacker.py'

        attacker_noise_mode='dxp'
        attacker_n_layers=(1)
        attacker_opt='adamw'
        attacker_lr=0.001
        attacker_epochs=20

        if [ "$attack_model" = "dec" ]; then
          attacker_n_layers=(1)
          attacker_lr=0.0001
          attacker_epochs=40
        fi
        if [ "$attack_model" = "gruattn" ]; then
          attacker_n_layers=(1)
          attacker_lr=0.0003
          attacker_epochs=30
        fi

        if [ "$attack_model" = "gruattn" ]; then
          attacker_n_layers=(1)
          attacker_lr=0.0001
          attacker_opt='adamw'
          attacker_epochs=50
        fi

        for attacker_n_layer in "${attacker_n_layers[@]}"; do
          attacker_prefix="${attacker_n_layer}layers-dxp${attacker_noise_scale_dxp}"

#          if [ "$attack_model" = "dec" ]; then
#            attacker_prefix="${attacker_n_layer}layers-dxp${noise_scale_dxp}"
#          fi


          echo "Running $file with seed=$seed, dataset=$attacker_dataset"
          python ../py/$file \
            --model_name "$model_name" \
            --seed "$seed" \
            --dataset "$attacker_dataset" \
            --attack_model "$attack_model" \
            --attack_mode "b2tr" \
            --sps "$sps" \
            --save_checkpoint True \
            --log_to_wandb False \
            --noise_mode "${attacker_noise_mode}" \
            --md_n_layers "$attacker_n_layer" \
            --epochs "$attacker_epochs" \
            --checkpoint_freq 10 \
            --require_prefix="$attacker_prefix" \
            --lr "$attacker_lr" \
            --opt "$attacker_opt" \
            --noise_scale_dxp "$attacker_noise_scale_dxp" \
            --dataset_train_frac "$atk_train_frac" \
            --dataset_test_frac 0.1

          case_name="${model_name}-${sfl_dataset}-${noise_mode}:${noise_scale_dxp}<${attack_model}${attacker_n_layer}-${attacker_dataset}"

          # 将其用于攻击
          echo "Running evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
          python ../py/evaluate_tag_methods.py \
            --noise_mode "$noise_mode" \
            --case_name "$case_name" \
            --model_name "$model_name" \
            --split_points "$sps" \
            --global_round "$global_round" \
            --seed "$seed" \
            --dataset "$sfl_dataset" \
            --noise_scale_dxp "$noise_scale" \
            --exp_name "$exp_name" \
            --attacker_b2tr_sp "$attacker_sp" \
            --attacker_tr2t_sp "$attacker_sp" \
            --attacker_prefix "$attacker_prefix" \
            --attacker_model "$attack_model" \
            --self_pt_enable "$self_pt_enable" \
            --client_num "$client_num" \
            --data_shrink_frac "$data_shrink_frac" \
            --test_data_shrink_frac "$test_data_shrink_frac" \
            --evaluate_freq "$evaluate_freq" \
            --client_steps "$client_steps" \
            --lora_at_top "$lora_at_top" \
            --lora_at_trunk "$lora_at_trunk" \
            --lora_at_bottom "$lora_at_bottom" \
            --collect_all_layers "$collect_all_layers" \
            --dataset_label "$dataset_label" \
            --attacker_dataset "$attacker_dataset" \
            --batch_size "$batch_size" \
            --dlg_enable "$dlg_enable" \
            --attacker_train_frac "$atk_train_frac" \
            --dlg_raw_enable "$dlg_raw_enable" \
            --attacker_freq "$attacker_freq" \
            --attacker_samples "$attacker_samples" \
            --max_global_step "$max_global_step"
        done
      done
    done
  done
done
