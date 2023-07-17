#!/bin/bash
python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 1 --exp_id test --batch_size 8 --ex_size 8 --lr 0.001 --weight_decay 0.01 --n_epochs 25 --skip_wandb --debug
