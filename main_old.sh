#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 1 --exp_id 001 --batch_size 8 --n_epochs 1 --skip_wandb --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 1 --exp_id 001 --batch_size 8 --ex_size 8 --n_epochs 1 --skip_wandb --debug 

