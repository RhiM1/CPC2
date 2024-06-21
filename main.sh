#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 000 --run_id 001 --N 1 --model minerva --ex_size 128 --p_factor 1 --skip_train --do_fit
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats XLSRFull --exp_id 000 --run_id 001 --N 1 --model minerva --ex_size 128 --p_factor 1 --skip_train --do_fit
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats WhisperFull --exp_id 000 --run_id 001 --N 1 --model minerva --ex_size 128 --p_factor 1 --skip_train --do_fit
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 001 --run_id 001 --N 1 --model ffnn_init --feat_embed_dim 512 --class_embed_dim 512 --lr 2e-2 --weight_decay 1e-4 --dropout 0 --n_epochs 100 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 002 --run_id 001 --N 1 --model minerva --ex_size 128 --feat_embed_dim 32 --class_embed_dim 1 --lr 2e-2 --weight_decay 1e-4 --p_factor 1 --n_epochs 100 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 003 --run_id 001 --N 1 --model minerva --ex_size 128 --feat_embed_dim 32 --class_embed_dim 1 --lr 2e-2 --lr_ex 2e-3 --weight_decay 1e-3 --wd_ex 1e-4 --p_factor 1 --n_epochs 100 --batch_size 128 --train_ex_class



