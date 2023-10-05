#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main_feature_explore.py --feats WhisperFull --model AttenPool_layers --N 1 --exp_id 001 --ex_size 1 --lr 0.0001 --weight_decay 0.001 --n_epochs 1 --exemplar_source all --skip_wandb --debug
