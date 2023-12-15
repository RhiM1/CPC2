#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats Spec --skip_wand --exp_id test --run_id 001 --N 1 --model minerva --ex_size 128 --fix_ex --p_factor 1 --skip_train
CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats WhisperFull --exp_id 002 --run_id 001b --skip_wandb --debug --N 1 --model minerva --ex_size 128 --fix_ex --feat_embed_dim 32 --lr 2e-3 --weight_decay 1e-3 --p_factor 1 --n_epochs 100 --batch_size 128
