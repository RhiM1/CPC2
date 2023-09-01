#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 1 --exp_id 007final --lr 0.00001 --weight_decay 0.0001 --train_disjoint --n_epochs 1 --skip_wandb --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 2 --exp_id 007final --lr 0.00001 --weight_decay 0.0001 --train_disjoint --n_epochs 1 --skip_wandb --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 3 --exp_id 007final --lr 0.00001 --weight_decay 0.0001 --train_disjoint --n_epochs 1 --skip_wandb --debug

#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 1 --exp_id 015final --ex_size 8 --lr 0.000002  --weight_decay 0.0001 --train_disjoint --n_epochs 1 --skip_wandb --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 2 --exp_id 015final --ex_size 8 --lr 0.000002  --weight_decay 0.0001 --train_disjoint --n_epochs 1 --skip_wandb --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 3 --exp_id 015final --ex_size 8 --lr 0.000002  --weight_decay 0.0001 --train_disjoint --n_epochs 1 --skip_wandb --debug

#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 1 --exp_id 021test2final --ex_size 8 --lr 0.000002 --weight_decay 0.00001 --train_disjoint --use_r_encoding --minerva_r_dim 4 --n_epochs 1 --skip_wandb --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 2 --exp_id 021test2final --ex_size 8 --lr 0.000002 --weight_decay 0.00001 --train_disjoint --use_r_encoding --minerva_r_dim 4 --n_epochs 1 --skip_wandb --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 3 --exp_id 021test2final --ex_size 8 --lr 0.000002 --weight_decay 0.00001 --train_disjoint --use_r_encoding --minerva_r_dim 4 --n_epochs 1 --skip_wandb --debug

CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id 001 --ex_size 8 --lr 0.0001 --weight_decay 0.001 --n_epochs 1 --skip_wandb --debug