# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[CCS]BiSR_diff_mia_split'
client_num=1
global_round=1
client_steps=500
noise_scale=0.0
noise_mode="none"
attacker_prefix='normal'
data_shrink_frac=0.08
test_data_shrink_frac=0.1
attacker_train_frac=0.5
evaluate_freq=1000
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

model_name='llama2'
attack_model='gru'
batch_size=2
attacker_freq=300
attacker_samples=5
max_global_step=605

wba_enable=False
wba_raw_enable=True
wba_lr=0.01
wba_epochs=180
wba_raw_epochs=2400

attacker_datasets=("piqa")
sfl_datasets=("piqa")
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")
sp1s=(2 4)

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    for sp1 in "${sp1s[@]}"; do
      sps="${sp1}-26"

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
        --dataset_train_frac "$attacker_train_frac" \
        --dataset_test_frac 0.1 \
        --save_checkpoint True \
        --log_to_wandb False

      case_name="${model_name}-${sfl_dataset}<${attacker_dataset}:${sps}"

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
        --attacker_b2tr_sp "$sp1" \
        --attacker_tr2t_sp "$sp1" \
        --attacker_prefix "$attacker_prefix" \
        --attacker_train_frac "$attacker_train_frac"\
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
        --dlg_enable False \
        --attacker_freq "$attacker_freq" \
        --attacker_samples "$attacker_samples" \
        --max_global_step "$max_global_step" \
        --wba_enable "$wba_enable" \
        --wba_epochs "$wba_epochs" \
        --wba_raw_enable "$wba_raw_enable" \
        --wba_raw_epochs "$wba_raw_epochs" \
        --wba_lr "$wba_lr"
    done
  done
done
