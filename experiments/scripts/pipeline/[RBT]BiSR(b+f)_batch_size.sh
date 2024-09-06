# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[RBT]BiSR(b+f)_batch_size'
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

sps="6-27"

attacker_freq=300
attacker_samples=5
max_global_step=605

sip_inverter_dataset='sensireplaced'

model_names=('gpt2-large')
sfl_datasets=("piqa-mini") #"piqa" "codealpaca" "dialogsum" "sensimarked"

batch_sizes=(8 7 6 5 4 3 2 1)

gma_lr=0.09
gma_beta=0.85
gma_epcs=(512 256 128 64 32)
gma_init_temp=1.2
gsma_lr=0.01
gsma_epcs=(800 400 200 100 50)
gsma_wd=0.01


pre_ft_dataset='codealpaca'
pre_ft_steps=(240000) #0 4800 9600 14400 18000

for batch_size in "${batch_sizes[@]}"; do
  for gma_epc in "${gma_epcs[@]}"; do
    for gsma_epc in "${gsma_epcs[@]}"; do
      for model_name in "${model_names[@]}"; do
        for sfl_dataset in "${sfl_datasets[@]}"; do
          case_name="BiSR(b+f)@${model_name}@${sfl_dataset}-${batch_size}-GMA${gma_epc}-SMA${gsma_epc}"

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
            --noise_scale "$noise_scale" \
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
            --gma_enable True \
            --gsma_enable False \
            --sma_enable True \
            --eia_enable False \
            --attacker_freq "$attacker_freq" \
            --attacker_samples "$attacker_samples" \
            --max_global_step "$max_global_step" \
            --sip_dataset "$sip_inverter_dataset" \
            --sip_prefix "normal" \
            --sip_b2tr_enable True \
            --sip_b2tr_layer -1 \
            --sip_tr2t_enable False \
            --gma_lr "$gma_lr" \
            --gma_beta "$gma_beta" \
            --gma_epochs "$gma_epc" \
            --gma_init_temp "$gma_init_temp" \
            --sma_lr "$gsma_lr" \
            --sma_epochs "$gsma_epc" \
            --sma_wd "$gsma_wd"
          #                        --gsma_lr "$gsma_lr" \
          #            --gsma_epochs "$gsma_epc" \
          #            --gsma_wd "$gsma_wd" \
        done
      done
    done
  done
done
