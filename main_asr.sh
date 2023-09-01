#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python whisper_asr.py --exp_id 004 --N 1 --lr 1e-5 --batch_size 8 --gradient_acc 2
#CUDA_VISIBLE_DEVICES=0 python whisper_asr.py --exp_id 004 --N 2 --lr 1e-5 --batch_size 8 --gradient_acc 2

CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 1 --exp_id 007 --lr 0.00001 --weight_decay 0.0001 --pretrained_feats_model whisper/004_1_lr1e-05_bs8_ga2/checkpoint-6000
