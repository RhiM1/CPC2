#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperBase --model LSTM_layers --N 1 --exp_id test --batch_size 2 --layer -1 --num_layers 6 --n_epochs 1 --debug --skip_wandb --save_feats
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM --N 1 --exp_id test --batch_size 2 --ex_size 8 --layer 11 --n_epochs 1 --debug --skip_wandb
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 2 --ex_size 8 --layer -1 --n_epochs 1 --debug --skip_wandb
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --N 1 --exp_id test --batch_size 2 --layer -1 --n_epochs 1 --debug --skip_wandb --save_feats
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 2 --ex_size 3 --layer -1 --n_epochs 1 --skip_wandb --debug --save_feats

