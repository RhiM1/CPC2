#!/bin/bash
python MAIN_predict_intel_correctness_feats.py --feats_model WhisperFull --model LSTM --exp_id 1_layer --batch_size 2 --layer 11 --n_epochs 10 --skip_wandb --weight_decay 0.001 --debug
python MAIN_predict_intel_correctness_feats.py --feats_model WhisperFull --model LSTM_layers --exp_id all_layer --batch_size 2 --layer -1 --n_epochs 10 --skip_wandb --weight_decay 0.001 --debug
