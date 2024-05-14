# 实验：如果客户端的Bottom和Top是事先微调过的
seed=42

exp_name='[EXP]SDRP_pre_ft_embed_lora'
client_num=1
global_round=1
client_steps=600
noise_scale=0.0
noise_mode="none"
attacker_prefix='normal'

test_data_shrink_frac=0.1
evaluate_freq=300
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=False
attack_model='gru'
batch_size=2

attacker_freq=200
attacker_samples=10

model_name='gpt2-large'
dataset_label='train'
data_shrink_frac=1.0 # 被攻击数据集的缩减比例
max_global_step=810  # 攻击800个batch
attacker_dataset="piqa"
attacker_training_fraction=1.0 # 攻击模型的训练集比例
sfl_dataset="piqa"

dlg_enable=True
dlg_adjust=0
dlg_epochs=18
dlg_beta=0.85
dlg_init_with_dra=True

pre_ft_dataset='codealpaca'
pre_ft_steps=(4800)

lora_at_embeds=(True False)

for lora_at_embed in "${lora_at_embeds[@]}"; do
  for pre_ft_max_steps in "${pre_ft_steps[@]}"; do

    sps="6-26"
    case_name="EMB-${lora_at_embed}:${model_name}-${sfl_dataset}-pre@${pre_ft_dataset}[${pre_ft_max_steps}]"

    # 先训练攻击模型
    echo "Running train_attacker.py with atk_ds=$attacker_dataset"
    python ../py/train_inverter.py \
      --model_name "$model_name" \
      --seed "$seed" \
      --dataset "$attacker_dataset" \
      --attack_model "$attack_model" \
      --attack_mode 'b2tr' \
      --noise_mode "$noise_mode" \
      --sps "$sps" \
      --save_checkpoint True \
      --log_to_wandb False

    echo "Running ${case_name} evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
    python ../py/sim_with_attacker.py \
      --noise_mode "$noise_mode" \
      --case_name "$case_name" \
      --model_name "$model_name" \
      --split_points "$sps" \
      --global_round "$global_round" \
      --seed "$seed" \
      --dataset "$sfl_dataset" \
      --noise_scale_dxp "$noise_scale" \
      --exp_name "$exp_name" \
      --attacker_model "$attack_model" \
      --attacker_b2tr_sp 6 \
      --attacker_tr2t_sp 6 \
      --attacker_prefix "$attacker_prefix" \
      --attacker_train_frac "$attacker_training_fraction" \
      --attacker_b2tr_enable True --attacker_tr2t_enable False --dlg_enable False \
      --self_pt_enable "$self_pt_enable" \
      --client_num "$client_num" \
      --data_shrink_frac "$data_shrink_frac" \
      --test_data_shrink_frac "$test_data_shrink_frac" \
      --evaluate_freq "$evaluate_freq" \
      --client_steps "$client_steps" \
      --lora_at_top "$lora_at_top" \
      --lora_at_trunk "$lora_at_trunk" \
      --lora_at_bottom "$lora_at_bottom" \
      --lora_at_embed "$lora_at_embed" \
      --collect_all_layers "$collect_all_layers" \
      --dataset_label "$dataset_label" \
      --attacker_dataset "$attacker_dataset" \
      --batch_size "$batch_size" \
      --attacker_freq "$attacker_freq" \
      --attacker_samples "$attacker_samples" \
      --max_global_step "$max_global_step" \
      --pre_ft_dataset "$pre_ft_dataset" \
      --pre_ft_max_steps "$pre_ft_max_steps" \
      --dlg_enable "$dlg_enable" \
      --dlg_adjust "$dlg_adjust" \
      --dlg_epochs "$dlg_epochs" \
      --dlg_beta "$dlg_beta" \
      --dlg_init_with_dra "$dlg_init_with_dra" \
      --dlg_raw_enable True

  done
done
doe
