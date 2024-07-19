export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u train.py \
  --root_path ../dataset/global/north/train \
  --data_path temp_train.npy \
  --model_id v1 \
  --model $model_name \
  --data Meteorology \
  --features MS \
  --seq_len 168 \
  --label_len 1 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 37 \
  --d_model 64 \
  --d_ff 64 \
  --n_heads 1 \
  --des 'global_temp' \
  --learning_rate 0.01 \
  --batch_size 20480 \
  --train_epochs 4 \
  --num_workers 0 \
  # --gpu 3
  # --devices '0,1'
