# 实验：要说明攻击模型对于噪声有适应作用
seeds=(42)

dataset_label='train'
test_data_label='test'
data_shrink_frac=0.06
test_data_shrink_frac=0.5

model_name='gpt2-large'

exp_name='[TEST]dra_noise_adaption'

client_num=1
client_steps=250
global_round=1
batch_size=2
dlg_enable=False

evaluate_freq=300
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
attacker_freq=100
attacker_samples=5
sps='6-26'
attacker_b2tr_sp=26
attacker_tr2t_sp=26

# 噪声规模
noise_modes=("gaussian")
sfl_noise_scales_gaussian=(0.02 0.01)
attacker_noises_gaussian=(0.01 0.02 0.0)
attacker_train_frac=0.3
noise_scale_dxp=0
attack_models=('gru')
datasets=('wikitext')

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for noise_mode in "${noise_modes[@]}"; do
      for attack_model in "${attack_models[@]}"; do
        for sfl_noise_scale_gaussian in "${sfl_noise_scales_gaussian[@]}"; do
          for noise_scale_gaussian in "${attacker_noises_gaussian[@]}"; do

            # 先训练攻击模型

            file='train_attacker.py'
            if [ "$attack_model" = "moe" ] || [ "$attack_model" = "moe2" ]; then
              file='train_attacker_moe.py'
            fi

            attack_mode='b2tr'
            if [ "$noise_mode" = "gaussian" ]; then
              attack_mode='tr2t'
            fi

            echo "Running $file with seed=$seed, dataset=$dataset, mode=$attack_mode"
            python ../py/$file \
              --model_name "$model_name" \
              --seed "$seed" \
              --dataset "$dataset" \
              --attack_model "$attack_model" \
              --attack_mode "$attack_mode" \
              --sps "$sps"\
              --save_checkpoint True \
              --log_to_wandb False \
              --noise_mode "$noise_mode" \
              --epochs 20\
              --noise_scale_dxp "$noise_scale_dxp" \
              --noise_scale_gaussian "$noise_scale_gaussian" \
              --dataset_train_frac "$attacker_train_frac"

            attacker_prefix="${noise_mode}:${noise_scale_gaussian}"
            case_name="${dataset}[${noise_mode}-${sfl_noise_scale_gaussian}]<ATK[${noise_mode}-${noise_scale_gaussian}]"

            echo "Running ${exp_name}"
            python ../py/evaluate_tag_methods.py \
              --noise_mode "$noise_mode" \
              --exp_name "$exp_name" \
              --model_name "$model_name" \
              --global_round "$global_round" \
              --seed "$seed" \
              --dataset "$dataset" \
              --dataset_label "$dataset_label" \
              --data_shrink_frac "$data_shrink_frac" \
              --test_data_label "$test_data_label" \
              --test_data_shrink_frac "$test_data_shrink_frac" \
              --case_name "$case_name" \
              --split_points "$sps" \
              --attacker_model "$attack_model" \
              --attacker_b2tr_sp "$attacker_b2tr_sp" \
              --attacker_tr2t_sp "$attacker_tr2t_sp" \
              --self_pt_enable "$self_pt_enable" \
              --client_num "$client_num" \
              --data_shrink_frac "$data_shrink_frac" \
              --evaluate_freq "$evaluate_freq" \
              --client_steps "$client_steps" \
              --lora_at_top "$lora_at_top" \
              --lora_at_trunk "$lora_at_trunk" \
              --lora_at_bottom "$lora_at_bottom" \
              --attacker_freq "$attacker_freq" \
              --attacker_samples "$attacker_samples" \
              --attacker_prefix "$attacker_prefix" \
              --attacker_train_frac "$attacker_train_frac" \
              --collect_all_layers False \
              --pre_ft_dataset "" \
              --dlg_enable "$dlg_enable" \
              --batch_size "$batch_size" \
              --noise_mode "$noise_mode" \
              --noise_scale_gaussian "$sfl_noise_scale_gaussian"
          done
        done
      done
    done
  done
done
