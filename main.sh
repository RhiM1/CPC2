#!/bin/bash


#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats_learn.py --feats WhisperFull --model LSTM_layers --layer -1 --N 1 --exp_id 001 --lr 0.00001 --weight_decay 0.0001
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats_learn.py --feats WhisperFull --model LSTM_layers --layer -1 --N 2 --exp_id 001 --lr 0.00001 --weight_decay 0.0001
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats_learn.py --feats WhisperFull --model LSTM_layers --layer -1 --N 3 --exp_id 001 --lr 0.00001 --weight_decay 0.0001

#CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats WhisperFull --model simple --N 1 --exp_id test --debug --skip_wandb --lr 0.000002 --weight_decay 0.0001 --n_epochs 1
CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats WhisperFull --model ex_simple --N 1 --exp_id test --debug --skip_wandb --fix_ex --ex_size 8 --feat_embed_dim 128 --lr 0.000002 --weight_decay 0.0001 --n_epochs 1 --skip_test

