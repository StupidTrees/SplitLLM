dataset='wikitext'
model_name="gpt2-large"
model='gru'
sip_inverter_dataset='wikitext'
load_bits=8
noise_mode='dxp'
noise_scale_dxp=0.2

echo "Running train_attacker.py with dataset=$dataset, model=$model"
echo "Running train_inverter.py"
python ../py/train_inverter.py \
  --model_name "$model_name" \
  --attack_model "gru" \
  --dataset "$sip_inverter_dataset" \
  --attack_mode 'b2tr' \
  --sps "6-22" \
  --noise_mode "$noise_mode" \
  --noise_scale_dxp "$noise_scale_dxp" \
  --dataset_test_frac 0.1 \
  --save_checkpoint True \
  --log_to_wandb False \
  --load_bits "$load_bits"
