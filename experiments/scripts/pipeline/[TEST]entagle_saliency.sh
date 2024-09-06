# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[TEST]ENTANGLE'
client_num=1
global_round=10
client_steps=500
noise_scale=0.2
noise_mode="none"
attacker_prefix='normal'
data_shrink_frac=0.08
test_data_shrink_frac=0.0005
evaluate_freq=800
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_names=('llama2')
attack_model='gru'
sps='6-26'
attacker_sp=6
batch_size=2

attacker_freq=800
attacker_samples=10
max_global_step=32000

attacker_datasets=("sensireplaced")
sfl_datasets=("piqa")
max_seq_len=-1
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    for model_name in "${model_names[@]}"; do
      dataset_test_frac=0.1

      if [ "$model_name" == "chatglm" ]; then
        max_seq_len=256
        dataset_test_frac=1.0
      fi

      if [ "$model_name" == "flan-t5-large" ]; then
        sps='6-20'
      fi

      # 先训练攻击模型
      echo "Running train_attacker.py with atk_ds=$attacker_dataset"
      python ../py/train_inverter.py \
        --model_name "$model_name" \
        --seed "$seed" \
        --dataset "$attacker_dataset" \
        --attack_model "$attack_model" \
        --attack_mode 'b2tr' \
        --sps "$sps" \
        --dataset_test_frac "$dataset_test_frac" \
        --save_checkpoint True \
        --log_to_wandb False
      #        --noise_mode "$noise_mode" \

      case_name="${model_name}-${sfl_dataset}<${attacker_dataset}"

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
        --dataset_max_seq_len "$max_seq_len" \
        --attacker_dataset "$attacker_dataset" \
        --batch_size "$batch_size" \
        --attacker_freq "$attacker_freq" \
        --attacker_samples "$attacker_samples" \
        --max_global_step "$max_global_step" \
        --noise_scale "$noise_scale"\
        --entangle_enable True
    done
  done
done
