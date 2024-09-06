
dataset_label='train'
global_round=1
client_steps=1000
noise_scale=0.0
noise_mode="none"
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=600
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=True

sps="6-27"
batch_size=6
attacker_freq=200
attacker_samples=5
max_global_step=605
sip_inverter_dataset='sensireplaced'

gma_lr=0.09
gma_beta=0.85
gma_epc=18
gma_init_temp=1.2
gsma_lr=0.005
gsma_epc=800
sma_epc=64
gsma_wd=0.02

exp_name='Example-BiSR'

model_names=('llama2')
sfl_datasets=("wikitext")
seeds=(42)

for seed in "${seeds[@]}"; do
  for model_name in "${model_names[@]}"; do
    for sfl_dataset in "${sfl_datasets[@]}"; do
      case_name="SD${seed}-BiSR(b+f)@${model_name}@${sfl_dataset}"

      launch Split Learning Simulation, while attacking the SL system
      echo "Running sim_with_attacker.py ..."
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
        --gsma_enable True \
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
        --gsma_lr "$gsma_lr" \
        --gsma_epochs "$gsma_epc" \
        --gsma_wd "$gsma_wd" \
        --sma_lr "$gsma_lr" \
        --sma_epochs "$sma_epc" \
        --sma_wd "$gsma_wd"
    done
  done
done
