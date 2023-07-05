#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py default_config --feats WhisperFull --model ExLSTM --exp_id test --batch_size 2 --layer 12 --p_factor 1 --n_epochs 1 --skip_wandb
CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py default_config --feats WhisperFull --model ExLSTM --exp_id test_loaded --batch_size 2 --layer 12 --p_factor 1 --n_epochs 1 --skip_wandb


#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py default_config --feats WhisperFull --model LSTM --exp_id 002 --batch_size 2 --layer 10 --n_epochs 1 --skip_wandb
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py default_config --feats WhisperFull --model ExLSTM --exp_id test --batch_size 2 --layer 10 --n_epochs 1 --skip_wandb --ex_size 8
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py default_config --feats WhisperEncoder --model ExLSTM --exp_id test --batch_size 2 --layer 11 --n_epochs 1 --skip_wandb --ex_size 8
