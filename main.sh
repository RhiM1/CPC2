#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats Spec --skip_wand --exp_id test --run_id 001 --N 1 --model minerva --ex_size 128 --fix_ex --p_factor 1 --skip_train
CUDA_VISIBLE_DEVICES=0 python MAIN_learn.py --feats WhisperFull --skip_wand --exp_id test --run_id 001 --N 1 --model minerva --ex_size 128 --fix_ex --p_factor 1 --skip_train
