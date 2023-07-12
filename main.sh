#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperBase --model LSTM_layers --N 1 --exp_id test --batch_size 2 --layer -1 --num_layers 6 --n_epochs 1 --debug --skip_wandb --save_feats --grad_clipping 1
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM --N 1 --exp_id test --batch_size 2 --ex_size 8 --layer 11 --n_epochs 1 --debug --skip_wandb
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --n_epochs 1 --debug --skip_wandb --minerva_dim 512
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layersClamp --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --n_epochs 1 --debug --skip_wandb --minerva_dim 512
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --N 1 --exp_id test --batch_size 2 --layer -1 --n_epochs 1 --debug --skip_wandb --save_feats
CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --n_epochs 1 --skip_wandb --debug --matching_exemplars
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_std --N 1 --exp_id test --batch_size 8 --ex_size 2 --layer -1 --n_epochs 1 --debug --skip_wandb --num_minervas 3 --random_exemplars

