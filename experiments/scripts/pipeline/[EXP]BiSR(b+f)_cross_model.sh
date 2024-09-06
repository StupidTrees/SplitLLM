# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[EXP]BiSR(b+f)_cross_model'
global_round=1
client_steps=500
noise_scale=0.0
noise_mode="none"
data_shrink_frac=0.08
test_data_shrink_frac=0.1
evaluate_freq=100
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True

sps="6-27"
batch_size=2

attacker_freq=200
attacker_samples=5
max_global_step=605

collect_all_layers=True

sfl_datasets=("piqa")
sfl_model_name='codegen'

sip_inverter_dataset='sensireplaced'
sip_inverter_models=('llama2')
sip_layer=6

gma_lr=0.09
gma_beta=0.85
gma_epc=18
gma_init_temp=1.2
gsma_lr=0.005
gsma_epc=800
gsma_wd=0.02
gsma_cross_model='llama2'
load_bits=32
# 0.05 0.001 0.1)

for sfl_dataset in "${sfl_datasets[@]}"; do
  for sip_inverter_model in "${sip_inverter_models[@]}"; do

    case_name="CM-${sfl_model_name}-${sfl_dataset}<<-${sip_inverter_model}-${sip_inverter_dataset}-l${sip_layer}-SCM${sma_cross_model}"

    # 先训练攻击模型
    echo "Running train_inverter.py"
    python ../py/train_inverter.py \
      --model_name "$sip_inverter_model" \
      --seed "$seed" \
      --attack_model "gru" \
      --dataset "$sip_inverter_dataset" \
      --attack_mode 'b2tr' \
      --sps "${sip_layer}-22" \
      --dataset_test_frac 0.1 \
      --save_checkpoint True \
      --log_to_wandb False\
      --load_bits "$load_bits"

    # 将其用于攻击
    echo "Running evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
    python ../py/sim_with_attacker.py \
      --noise_mode "$noise_mode" \
      --case_name "$case_name" \
      --model_name "$sfl_model_name" \
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
      --attacker_freq "$attacker_freq" \
      --attacker_samples "$attacker_samples" \
      --max_global_step "$max_global_step" \
      --sip_dataset "$sip_inverter_dataset" \
      --tag_enable False \
      --gma_enable True \
      --gsma_enable True \
      --sma_enable False \
      --eia_enable False \
      --sip_prefix "normal" \
      --sip_b2tr_enable True \
      --sip_b2tr_layer "$sip_layer" \
      --sip_tr2t_enable False \
      --sip_attack_all_layers False \
      --sip_target_model_name "$sip_inverter_model" \
      --gma_beta "$gma_beta" \
      --gma_epochs "$gma_epc" \
      --gma_init_temp "$gma_init_temp" \
      --gma_lr "$gma_lr" \
      --gsma_lr "$gsma_lr" \
      --gsma_epochs "$gsma_epc" \
      --gsma_wd "$gsma_wd" \
      --gsma_cross_model "$gsma_cross_model"\
      --load_bits "$load_bits"
  done
done
