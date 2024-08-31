#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python MAIN_thesis_combine_adapt.py --feats Spec --exp_id adapt --run_id test --skip_wandb --n_epochs 1 --N 1 --model minerva --ex_size 128 128 --feat_embed_dim 32 32 --class_embed_dim 1 --lr 2e-2 2e-2 --weight_decay 1e-4 1e-4 --p_factor 1 5 --dropout 0 0 --batch_size 128 --seed 42 42

exit

CUDA_VISIBLE_DEVICES=0 python MAIN_thesis_combine.py --feats WhisperFull --exp_id 002u --run_id 001 001 001 001 --N 1 --model minerva --ex_size 128 128 128 128 --feat_embed_dim 32 32 32 32 --class_embed_dim 1 --lr 2e-3 2e-3 2e-3 2e-3 --weight_decay 1e-3 1e-3 1e-3 1e-3 --p_factor 3 3 3 3 --dropout 0.2 0.2 0.2 0.2 --batch_size 128 --seed 42 84 168 336 --n_epochs 100 
exit

CUDA_VISIBLE_DEVICES=0 python MAIN_thesis_combine.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 006 --run_id 8 16 32 64 --N 1 --model minerva --ex_size 128 128 128 128 --fix_ex --feat_embed_dim 8 16 32 64 --class_embed_dim 1 --lr 2e-2 2e-2 2e-2 2e-2 --weight_decay 1e-4 1e-4 1e-4 1e-4 --p_factor 1 1 1 1  --dropout 0 0 0 0 --batch_size 128 --seed 1234 1234 1234 1234

exit

CUDA_VISIBLE_DEVICES=0 python MAIN_CSL_repeat.py --feats Spec --exp_id 004 --run_id 001 --skip_wandb --N 1 --model minerva --ex_size 128 --p_factor 1 --skip_train --do_fit
exit

CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 000 --run_id 001 --N 1 --model minerva --ex_size 128 --p_factor 1 --skip_train --do_fit
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats XLSRFull --exp_id 000 --run_id 001 --N 1 --model minerva --ex_size 128 --p_factor 1 --skip_train --do_fit
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats WhisperFull --exp_id 000 --run_id 001 --N 1 --model minerva --ex_size 128 --p_factor 1 --skip_train --do_fit
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 001 --run_id 001 --N 1 --model ffnn_init --feat_embed_dim 512 --class_embed_dim 512 --lr 2e-2 --weight_decay 1e-4 --dropout 0 --n_epochs 100 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 002 --run_id 001 --N 1 --model minerva --ex_size 128 --feat_embed_dim 32 --class_embed_dim 1 --lr 2e-2 --weight_decay 1e-4 --p_factor 1 --n_epochs 100 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python MAIN_CSL.py --skip_wandb --n_epochs 1 --feats Spec --exp_id 003 --run_id 001 --N 1 --model minerva --ex_size 128 --feat_embed_dim 32 --class_embed_dim 1 --lr 2e-2 --lr_ex 2e-3 --weight_decay 1e-3 --wd_ex 1e-4 --p_factor 1 --n_epochs 100 --batch_size 128 --train_ex_class



