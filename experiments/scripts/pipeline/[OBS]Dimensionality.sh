# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[OBS]Dimensionality'
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

batch_size=2

attacker_freq=200
attacker_samples=1
max_global_step=2405
reducer_train_frac=1.0
inverter_dataset="piqa"
sfl_dataset="piqa"

reducer_alphas=(3072)
split_point=6
sps="$split_point-27"

# 先训练Reducer

for alpha in "${reducer_alphas[@]}"; do

  echo "Running train_reducer.py with seed=$seed"
  python ../py/train_reducer.py \
    --model_name "$model_name" \
    --seed "$seed" \
    --dataset "$sfl_dataset" \
    --attack_mode "b2tr" \
    --layer "$split_point" \
    --alpha "$alpha" \
    --save_checkpoint True \
    --log_to_wandb False \
    --epochs 20 \
    --dataset_train_frac "$reducer_train_frac" \
    --dataset_train_label "train" \
    --dataset_test_frac 0.05 --checkpoint_freq 1

  # 再训练Inverter
  echo "Running train_inverter.py"
  python ../py/train_inverter.py \
    --model_name "$model_name" \
    --seed "$seed" \
    --attack_model "gru" \
    --dataset "$inverter_dataset" \
    --attack_mode 'b2tr' \
    --sps "$sps" \
    --epochs 20 \
    --dataset_test_frac 0.1 \
    --save_checkpoint True \
    --log_to_wandb False \
    --require_prefix "red:${alpha}" \
    --reducer_dataset "$sfl_dataset" \
    --reducer_train_frac "$reducer_train_frac" \
    --reducer_alpha "$alpha"

  case_name="DIM@${model_name}-inv${inverter_dataset}-red${alpha}${alpha}"

  # 将其用于攻击
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
    --self_pt_enable "$self_pt_enable" \
    --client_num "1" \
    --data_shrink_frac "$data_shrink_frac" \
    --test_data_shrink_frac "$test_data_shrink_frac" \
    --evaluate_freq "$evaluate_freq" \
    --client_steps "$client_steps" \
    --lora_at_top "$lora_at_top" \
    --lora_at_trunk "$lora_at_trunk" \
    --lora_at_bottom "$lora_at_bottom" \
    --collect_all_layers "$collect_all_layers" \
    --dataset_label "$dataset_label" \
    --batch_size "$batch_size" \
    --sip_enable True \
    --sip_b2tr_enable True \
    --sip_tr2t_enable False \
    --sip_b2tr_layer -1 \
    --sip_dataset "$inverter_dataset" \
    --sip_prefix "red:${alpha}" \
    --tag_enable False \
    --gma_enable False \
    --gsma_enable False \
    --sma_enable False \
    --eia_enable False \
    --attacker_freq "$attacker_freq" \
    --attacker_samples "$attacker_samples" \
    --max_global_step "$max_global_step" \
    --reducer_enable True \
    --reducer_alpha "$alpha" \
    --reducer_dataset "$sfl_dataset" \
    --reducer_layer "$split_point" \
    --reducer_train_frac "$reducer_train_frac"
done
