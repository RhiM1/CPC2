#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM --N 1 --exp_id test --batch_size 2 --layer 11 --n_epochs 1 --debug
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM --N 1 --exp_id 001 --batch_size 2 --layer 11 --n_epochs 20

