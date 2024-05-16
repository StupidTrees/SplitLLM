# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[HPT]BiSR(f)'
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

model_name='chatglm'

sps="6-27"
batch_size=2

attacker_freq=200
attacker_samples=1
max_global_step=405

sfl_datasets=("piqa")
sip_inverter_dataset='sensireplaced'

# chatglm
sma_lrs=(0.01 0.005 0.0001)
sma_epochs=(100 200 600 800)
sma_wds=(0.01 0.001 0.02)

# LLaMA2
#sma_lrs=(0.001 0.005 0.0001)
#sma_epochs=(100 300 1000)
#sma_wds=(0.01 0.001 0.02)

# 0.05 0.001 0.1)

for sfl_dataset in "${sfl_datasets[@]}"; do
  for sma_lr in "${sma_lrs[@]}"; do
    for sma_epc in "${sma_epochs[@]}"; do
      for sma_wd in "${sma_wds[@]}"; do
        case_name="BiSR(f)@${model_name}@${sfl_dataset},lr=${sma_lr},epc=${sma_epc},wd=${sma_wd}"

        # 先训练攻击模型
        echo "Running train_inverter.py"
        python ../py/train_inverter.py \
          --model_name "$model_name" \
          --seed "$seed" \
          --attack_model "gru" \
          --dataset "$sip_inverter_dataset" \
          --attack_mode 'b2tr' \
          --sps "$sps" \
          --dataset_test_frac 0.1 \
          --save_checkpoint True \
          --log_to_wandb False

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
          --noise_scale_dxp "$noise_scale" \
          --exp_name "$exp_name" \
          --self_pt_enable "$self_pt_enable" \
          --client_num 1 \
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
          --tag_enable False \
          --sma_enable True \
          --gsma_enable False \
          --gma_enable False \
          --eia_enable False --attacker_freq "$attacker_freq" \
          --attacker_samples "$attacker_samples" \
          --max_global_step "$max_global_step" \
          --sip_dataset "$sip_inverter_dataset" \
          --sip_prefix "normal" \
          --sip_b2tr_enable True \
          --sip_b2tr_layer -1 \
          --sip_tr2t_enable False \
          --sma_lr "$sma_lr" \
          --sma_epochs "$sma_epc" \
          --sma_wd "$sma_wd"
      done
    done
  done
done
