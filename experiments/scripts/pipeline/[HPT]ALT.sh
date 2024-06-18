# 实验：对Embedding Inversion Attack进行超参搜索
seed=42

dataset_label='train'
exp_name='[HPT]ALT'
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

sps="6-27"
batch_size=2

attacker_freq=200
attacker_samples=1
max_global_step=405

sfl_datasets=("piqa")
sip_inverter_dataset='sensireplaced'

alt_epochs=(1 5)


eia_lrs=(0.09 0.01 0.005)
eia_epochs=(100 38)
eia_temps=(0.3)
eia_wds=(0.01)
gma_lrs=(0.09 0.03)
gma_betas=(0.85)
gma_epochs=(18 64)
gma_init_temp=(1.0)


for sfl_dataset in "${sfl_datasets[@]}"; do
  for alt_epc in "${alt_epochs[@]}"; do
    for gma_lr in "${gma_lrs[@]}"; do
      for gma_epc in "${gma_epochs[@]}"; do
        for gma_beta in "${gma_betas[@]}"; do
          for gma_init_temp in "${gma_init_temp[@]}"; do
            for eia_lr in "${eia_lrs[@]}"; do
              for eia_epc in "${eia_epochs[@]}"; do
                for eia_temp in "${eia_temps[@]}"; do
                  for eia_wd in "${eia_wds[@]}"; do

                    # 先训练Inverter模型和Mapper
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

                    echo "Running train_mapper.py with seed=$seed"
                    python ../py/train_mapper.py \
                      --model_name "$model_name" \
                      --seed "$seed" \
                      --dataset "$sip_inverter_dataset" \
                      --attack_mode "b2tr" \
                      --target "6-1" \
                      --save_checkpoint True \
                      --log_to_wandb False \
                      --epochs 10 \
                      --dataset_train_frac 1.0 \
                      --dataset_test_frac 0.1

                    case_name="ALT@${model_name}@${sfl_dataset},epochs=${alt_epc},b_lr=${gma_lr},b_beta=${gma_beta},b_epochs=${gma_epc},b_temp=${gma_init_temp},f_lr=${eia_lr},f_epochs=${eia_epc},f_temp=${eia_temp},f_wd=${eia_wd}"

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
                      --sip_b2tr_enable False \
                      --sip_tr2t_enable False \
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
                      --tag_enable False \
                      --gma_enable False \
                      --gsma_enable False \
                      --sma_enable False \
                      --eia_enable False \
                      --sip_dataset "$sip_inverter_dataset" \
                      --sip_prefix "normal" \
                      --sip_b2tr_enable True \
                      --sip_b2tr_layer -1 \
                      --sip_tr2t_enable False \
                      --attacker_freq "$attacker_freq" \
                      --attacker_samples "$attacker_samples" \
                      --max_global_step "$max_global_step" \
                      --alt_enable True \
                      --alt_epochs "${alt_epc}" \
                      --alt_b_epochs "${gma_epc}" \
                      --alt_b_lr "${gma_lr}" \
                      --alt_b_beta "${gma_beta}" \
                      --alt_b_init_temp "${gma_init_temp}" \
                      --alt_f_epochs "${eia_epc}" \
                      --alt_f_lr "${eia_lr}" \
                      --alt_f_temp "${eia_temp}" \
                      --alt_f_wd "${eia_wd}" \
                      --eia_mapper_dataset "sensireplaced"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
