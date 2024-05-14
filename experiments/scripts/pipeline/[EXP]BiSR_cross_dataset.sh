# 实验：跨层DRA攻击
seed=42

dataset_label='train'
exp_name='[CCS]BIG_TABLE2'
client_num=1
global_round=1
client_steps=500
noise_scale=0.2
noise_mode="none"
attacker_prefix='normal'
data_shrink_frac=0.08
test_data_shrink_frac=0.3
evaluate_freq=1000
self_pt_enable=False
lora_at_trunk=True
lora_at_bottom=True
lora_at_top=True
collect_all_layers=False

model_names=('chatglm')
attack_model='gru'
sps='6-26'
attacker_sp=6
batch_size=2
dlg_enable=False
dlg_adjust=0
dlg_epochs=18
dlg_beta=0.85
dlg_lr=0.09
dlg_init_with_dra=True
dlg_raw_enable=False
dlg_raw_epochs=400
dlg_method='tag'
dlg_lamp_freq=30

wba_enable=False
wba_raw_enable=False
wba_lr=0.001
wba_raw_epochs=1000
wba_epochs=400
wba_dir_enable=True

alt_enable=False
alt_steps=2
alt_fwd_steps=64
alt_bwd_steps=18

attacker_freq=200
attacker_samples=10
max_global_step=610


attacker_datasets=("sensireplaced")
sfl_datasets=( "codealpaca"  "sensimarked")

max_seq_len=-1
#("piqa" "codealpaca" "dialogsum"  "sensimarked" "gsm8k" "wikitext")

for attacker_dataset in "${attacker_datasets[@]}"; do
  for sfl_dataset in "${sfl_datasets[@]}"; do
    for model_name in "${model_names[@]}"; do
      dataset_test_frac=0.1
      if [ "$model_name" == "llama2" ] || [ "$model_name" == "llama3" ]; then
        dlg_epochs=6
        wba_lr=0.001
        wba_epochs=160
        wba_raw_epochs=2400
        if [ "$sfl_dataset" == "codealpaca" ]; then
          dlg_epochs=30
        fi
        if [ "$sfl_dataset" == "wikitext" ]; then
          dlg_epochs=30
        fi
        if [ "$dlg_method" == "bisr" ]; then
          dlg_epochs=18
          dlg_raw_enable=False
          dlg_lamp_freq=6
        fi
      fi

      #      if [ "$model_name" == "llama3" ]; then
      #        max_seq_len=256
      #      fi

      if [ "$model_name" == "chatglm" ]; then
        dlg_epochs=18
        max_seq_len=256
        dlg_raw_epochs=500
        dataset_test_frac=1.0
        wba_lr=0.01
        wba_epochs=500
        wba_raw_epochs=2400
      fi

      if [ "$model_name" == "gpt2-large" ]; then
        dlg_epochs=18
        wba_lr=0.01
        wba_epochs=600
        wba_raw_epochs=2400
        if [ "$sfl_dataset" == "codealpaca" ]; then
          wba_lr=0.01
          wba_epochs=200
          wba_raw_epochs=2400
        fi
      fi

      if [ "$model_name" == "flan-t5-large" ]; then
        dlg_epochs=30
        sps='6-20'
      fi

      if [ "$dlg_method" == "lamp" ]; then
        dlg_epochs=400
        dlg_lamp_freq=50
        dlg_raw_enable=False
      fi

      tag="$dlg_method"

      if [ "$wba_enable" == "True" ]; then
#        dlg_enable=True
        dlg_raw_enable=False
        tag="wba"
      fi



      # !!!!
      dlg_raw_enable=False
      wba_raw_enable=False

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

      case_name="${model_name}-${sfl_dataset}<${attacker_dataset}-[${tag}]"

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
        --dlg_enable "$dlg_enable" \
        --dlg_adjust "$dlg_adjust" \
        --dlg_epochs "$dlg_epochs" \
        --dlg_beta "$dlg_beta" \
        --dlg_lamp_freq "$dlg_lamp_freq" \
        --dlg_method "$dlg_method" \
        --dlg_lr "$dlg_lr" \
        --dlg_init_with_dra "$dlg_init_with_dra" \
        --dlg_raw_enable "$dlg_raw_enable" \
        --dlg_raw_epochs "$dlg_raw_epochs" \
        --attacker_freq "$attacker_freq" \
        --attacker_samples "$attacker_samples" \
        --max_global_step "$max_global_step" \
        --wba_enable "$wba_enable" \
        --wba_epochs "$wba_epochs" \
        --wba_raw_enable "$wba_raw_enable" \
        --wba_raw_epochs "$wba_raw_epochs" \
        --noise_scale_dxp "$noise_scale" \
        --alt_enable "$alt_enable" \
        --alt_steps "$alt_steps" \
        --alt_fwd_steps "$alt_fwd_steps" \
        --alt_bwd_steps "$alt_bwd_steps" \
        --wba_lr "$wba_lr"\
        --wba_dir_enable "$wba_dir_enable"
    done
  done
done
