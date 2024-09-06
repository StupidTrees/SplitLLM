# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[EXP]BiSR_diff_noise_dxp'
client_num=1
global_round=1
client_steps=500
noise_scale=0.0
noise_mode="none"
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
dlg_adjust=0
dlg_epochs=18
dlg_raw_epochs=300
dlg_lr=0.001
dlg_beta=0.85
dlg_init_with_dra=True
dlg_method='tag'
attacker_freq=200
attacker_samples=10
max_global_step=610
atk_train_frac=1.0

noise_mode='dxp'
noise_scales=(0.1 0.15 0.2 0.25 0.3 0.35 0.4)
attack_models=('moe')

attacker_datasets=("sensireplaced")
sfl_datasets=("piqa")
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")

if [ "$model_name" = "llama2" ]; then
  dlg_epochs=200
  dlg_lr=0.08
fi

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    for noise_scale in "${noise_scales[@]}"; do
      for attack_model in "${attack_models[@]}"; do
        noise_scale="$noise_scale"
        # 先训练攻击模型

        file='train_inverter.py'
        if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
          file='train_inverter_moe.py'
        fi
        if [ "$attack_model" = "nop" ]; then
          file='train_inverter_no_pretrained.py'
        fi

        dlg_enable=True
        dlg_raw_enable=True
        attacker_noise_mode='dxp'
        if [ "$attack_model" = "gru" ]; then
          attacker_noise_mode='none'
          dlg_raw_enable=False
        fi

        attacker_prefix="normal"
        if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
          attacker_prefix="${attacker_noise_mode}"
        fi


        if [ "$attack_model" = "nop" ]; then
          attacker_prefix="nop"
          attack_model="gru"
          attacker_noise_mode='none'
          dlg_enable=False
        fi

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
          --epochs 20 \
          --noise_scale "$noise_scale" \
          --dataset_train_frac "$atk_train_frac" \
          --dataset_test_frac 0.1

        case_name="${model_name}-${sfl_dataset}-${noise_mode}:${noise_scale}<${attack_model}-${attacker_dataset}[${attacker_prefix}]"

        # 将其用于攻击
        echo "Running evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
        python ../py/sim_with_attacker.py \
          --noise_mode "$noise_mode" \
          --case_name "$case_name" \
          --model_name "$model_name" \
          --split_points "$sps" \
          --global_round "$global_round" \
          --seed "$seed" \
          --dataset "$sfl_dataset" \
          --noise_scale "$noise_scale" \
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
          --dlg_raw_epochs "$dlg_raw_epochs"\
          --dlg_beta "$dlg_beta" \
          --dlg_lr "$dlg_lr"\
          --dlg_method "$dlg_method" \
          --attacker_train_frac "$atk_train_frac" \
          --dlg_init_with_dra "$dlg_init_with_dra" \
          --dlg_raw_enable "$dlg_raw_enable" \
          --attacker_freq "$attacker_freq" \
          --attacker_samples "$attacker_samples" \
          --max_global_step "$max_global_step"
      done
    done
  done
done
