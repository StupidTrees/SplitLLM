# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[CCS]BiSR_diff_noise_gaussian'
client_num=1
global_round=1
client_steps=500
attacker_prefix='normal'
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=300
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='gpt2-large'

sps='6-26'
attacker_sp=26
batch_size=2
dlg_enable=True
dlg_adjust=0
dlg_epochs=18
dlg_beta=0.85
dlg_lr=0.08
dlg_init_with_dra=True
dlg_raw_enable=True
attacker_freq=200
attacker_samples=5
max_global_step=410
wba_enable=True

wba_lr=0.01
wba_epochs=100
wba_raw_epochs=1600
wba_dir_enable=True

noise_mode='gaussian'
noise_scale_gaussians=(2.0 1.75 1.5 1.25 1.0 0.75 0.5)
attack_models=('moe')

attacker_datasets=("piqa")
sfl_datasets=("piqa")
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    for noise_scale_gaussian in "${noise_scale_gaussians[@]}"; do
      raw_tested=False

      for attack_model in "${attack_models[@]}"; do
        # 先训练攻击模型

        dlg_raw_enable=True
        dlg_enable=True
        wba_raw_enable=False
        wba_enable=True
        file='train_attacker.py'
        if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
          file='train_attacker_moe.py'
        fi
        if [ "$attack_model" = "nop" ]; then
          file='train_attacker_no_pretrained.py'
        fi

        if [ "$raw_tested" = "True" ]; then
          dlg_raw_enable=False
          wba_raw_enable=False
        fi
        if [ "$raw_tested" = "False" ]; then
          raw_tested=True
        fi

        attacker_noise_mode='gaussian'
        if [ "$attack_model" = "gru" ]; then
          attacker_noise_mode='none'
          dlg_enable=False
          wba_enable=False
          wba_raw_enable=False
        fi

        atk_save_freq=5
        atk_save_threshold=0.1
        attacker_prefix="normal"
        if [ "$attack_model" = "nop" ]; then
          attacker_prefix="nop"
          attack_model="gru"
          attacker_noise_mode='none'
          dlg_enable=False
          atk_save_freq=1
          atk_save_threshold=0.01
        fi

        echo "Running $file with seed=$seed, dataset=$attacker_dataset"
        python ../py/$file \
          --model_name "$model_name" \
          --seed "$seed" \
          --dataset "$attacker_dataset" \
          --attack_model "$attack_model" \
          --attack_mode "tr2t" \
          --sps "$sps" \
          --save_checkpoint True \
          --log_to_wandb False \
          --noise_mode "${attacker_noise_mode}" \
          --epochs 20 \
          --noise_scale_gaussian "$noise_scale_gaussian" \
          --dataset_train_frac 1.0\
          --checkpoint_freq "$atk_save_freq" \
          --save_threshold "$atk_save_threshold" \
          --dataset_test_frac 0.1


        if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
          attacker_prefix="${attacker_noise_mode}"
        fi

        #!!!
        wba_raw_enable=False
        dlg_raw_enable=False

        case_name="${model_name}-${sfl_dataset}-${noise_mode}:${noise_scale_gaussian}<${attack_model}-${attacker_dataset}"

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
          --noise_scale_gaussian "$noise_scale_gaussian" \
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
          --dlg_adjust "$dlg_adjust" \
          --dlg_epochs "$dlg_epochs" \
          --dlg_beta "$dlg_beta" \
          --dlg_init_with_dra "$dlg_init_with_dra" \
          --dlg_raw_enable "$dlg_raw_enable" \
          --attacker_freq "$attacker_freq" \
          --attacker_samples "$attacker_samples" \
          --max_global_step "$max_global_step" \
          --dlg_lr "$dlg_lr" \
          --wba_enable "$wba_enable" \
          --wba_raw_enable "$wba_raw_enable" \
          --wba_lr "$wba_lr" \
          --wba_raw_epochs "$wba_raw_epochs" \
          --wba_epochs "$wba_epochs"\
          --wba_at tr2t\
          --wba_dir_enable "$wba_dir_enable"
      done
    done
  done
done
