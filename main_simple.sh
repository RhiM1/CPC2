#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats WhisperFull --exp_id 001 --run_id 001 --N 1 --model simple --lr 2e-3 --weight_decay 1e-3 --n_epochs 100
CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats WhisperFull --exp_id 001 --run_id 001 --N 2 --model simple --lr 2e-3 --weight_decay 1e-3 --n_epochs 100
CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats WhisperFull --exp_id 001 --run_id 001 --N 3 --model simple --lr 2e-3 --weight_decay 1e-3 --n_epochs 100

