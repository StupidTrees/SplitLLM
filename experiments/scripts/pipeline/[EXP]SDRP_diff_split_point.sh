# 实验：不同模型的不同层攻击难度
seed=42

exp_name='[EXP]SDRP_diff_split_point'
client_num=1
global_round=1
client_steps=600
noise_scale=0.0
noise_mode="none"
attacker_prefix='normal'

test_data_shrink_frac=0.1
evaluate_freq=3000
attacker_freq=200
attacker_samples=20
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True
attack_model='gru'
batch_size=2

model_names=('bert-large' 'llama2' 'flan-t5-large' 'roberta-large'  'gpt2-large')

dataset_label='train'
data_shrink_frac=1.0           # 被攻击数据集的缩减比例
max_global_step=1200            # 攻击1100个样本

for model_name in "${model_names[@]}"; do

  split_points=()
  attacker_dataset="piqa"
  attacker_training_fraction=1.0 # 攻击模型的训练集比例
  attacker_test_fraction=0.1     # 攻击模型的测试集比例
  sfl_dataset="piqa"

  if [ "$model_name" = 'llama2' ] || [ "$model_name" = 'gpt2-large' ]; then
    split_points=(3 6 9 12 15 18 21 24 27 30)
  fi
  if [ "$model_name" = 'flan-t5-large' ]; then
    split_points=(3 6 9 12 15 18 21)
  fi
  if [ "$model_name" = 'bert-large' ] || [ "$model_name" = 'roberta-large' ]; then
    attacker_dataset="imdb"
    attacker_training_fraction=0.015 # 攻击模型的训练集比例
    attacker_test_fraction=0.002
    sfl_dataset="imdb"
    split_points=(3 6 9 12 15 18 21)
  fi

  for sp in "${split_points[@]}"; do

    sps="${sp}-20"
    case_name="${model_name}-${sp}"

    # 先训练攻击模型
    echo "Running ${case_name} train_attacker.py with atk_ds=${attacker_dataset}"
    python ../py/train_attacker.py \
      --model_name "$model_name" \
      --seed "$seed" \
      --dataset "$attacker_dataset" \
      --attack_model "$attack_model" \
      --attack_mode 'b2tr' \
      --noise_mode "$noise_mode" \
      --sps "$sps" \
      --dataset_train_frac "$attacker_training_fraction" \
      --dataset_test_frac "$attacker_test_fraction" \
      --save_checkpoint True \
      --checkpoint_freq 10 \
      --epochs 20 \
      --log_to_wandb False

    # 将其用于攻击
    echo "Running ${case_name} evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
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
      --attacker_b2tr_sp "$sp" \
      --attacker_tr2t_sp "$sp" \
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
      --collect_all_layers "$collect_all_layers" \
      --dataset_label "$dataset_label" \
      --attacker_dataset "$attacker_dataset" \
      --batch_size "$batch_size" \
      --attacker_freq "$attacker_freq" \
      --attacker_samples "$attacker_samples" \
      --max_global_step "$max_global_step"

  done

done
