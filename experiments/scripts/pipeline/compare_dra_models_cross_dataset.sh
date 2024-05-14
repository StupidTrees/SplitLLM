# 实验：跨层黑盒攻击
seed=42

dataset_label='train'
test_data_label='test'
model_name='gpt2-large'
sp='6-30'
exp_name='[TEST]dra_models_cross_dataset'
client_num=1
global_round=1
client_steps=250
noise_dxp=0.0
noise_grad=0.0
noise_mode="none"
attacker_freq=200
attacker_samples=5
attacker_sp=6
data_shrink_frac=0.10
test_data_shrink_frac=0.5
evaluate_freq=500
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
dlg_enable=True
dlg_beta=0.85
dlg_init_with_dra=True
dlg_raw_enable=True
dlg_epoch=30
batch_size=2

attacker_dataset="dialogsum,wikitext,gsm8k"
# 观察不同的模型
sfl_datasets=("codealpaca" "piqa" "sanitized")
attacker_models=('moe2' 'gru')

for sfl_dataset in "${sfl_datasets[@]}"; do
  for attacker_model in "${attacker_models[@]}"; do
    case_name="${sfl_dataset}<${attacker_dataset}-${attacker_model}>"

    # 将其用于攻击
    echo "Running evaluate_tag_methods.py with sfl_ds=$sfl_dataset"
    python ../py/sim_with_attacker.py \
      --noise_mode "$noise_mode" \
      --case_name "$case_name" \
      --model_name "$model_name" \
      --global_round "$global_round" \
      --seed "$seed" \
      --dataset "$sfl_dataset" \
      --dataset_label "$dataset_label" \
      --data_shrink_frac "$data_shrink_frac" \
      --test_data_label "$test_data_label" \
      --test_data_shrink_frac "$test_data_shrink_frac" \
      --exp_name "$exp_name" \
      --split_points "$sp" \
      --attacker_model "$attacker_model" \
      --attacker_b2tr_sp "$attacker_sp" \
      --attacker_tr2t_sp "$attacker_sp" \
      --attacker_dataset "$attacker_dataset" \
      --self_pt_enable "$self_pt_enable" \
      --client_num "$client_num" \
      --data_shrink_frac "$data_shrink_frac" \
      --evaluate_freq "$evaluate_freq" \
      --client_steps "$client_steps" \
      --lora_at_top "$lora_at_top" \
      --lora_at_trunk "$lora_at_trunk" \
      --lora_at_bottom "$lora_at_bottom" \
      --attacker_freq "$attacker_freq" \
      --attacker_samples "$attacker_samples" \
      --collect_all_layers False \
      --dlg_enable "$dlg_enable" \
      --dlg_init_with_dra "$dlg_init_with_dra" \
      --dlg_epochs "$dlg_epoch" \
      --dlg_raw_enable "$dlg_raw_enable" \
      --dlg_beta "$dlg_beta" \
      --batch_size "$batch_size" \
      --noise_mode "$noise_mode" \
      --noise_scale_dxp "$noise_dxp" \
      --noise_scale_grad "$noise_grad"
  done
done
