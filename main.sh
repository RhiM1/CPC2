#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperBase --model LSTM_layers --N 1 --exp_id test --batch_size 2 --layer -1 --num_layers 6 --n_epochs 1 --debug --skip_wandb --save_feats --grad_clipping 1
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM --N 1 --exp_id test --batch_size 2 --ex_size 8 --layer 11 --n_epochs 1 --debug --skip_wandb
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --n_epochs 1 --debug --skip_wandb --minerva_dim 512
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layersClamp --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --n_epochs 1 --debug --skip_wandb --minerva_dim 512
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --N 1 --exp_id test --batch_size 2 --layer -1 --n_epochs 1 --debug --skip_wandb --save_feats
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --n_epochs 1 --skip_wandb --debug --matching_exemplars
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_std --N 1 --exp_id test --batch_size 8 --ex_size 2 --layer -1 --n_epochs 1 --debug --skip_wandb --num_minervas 3 --random_exemplars

#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 8 --ex_size 3 --layer -1 --n_epochs 1 --weight_decay 0.005 --lr 0.001 --grad_clipping 1 --minerva_dim 512 --skip_wandb --debug --num_minervas 3 --skip_train --pretrained_model_dir test_1_WhisperFull_ExLSTM_layers_09-36-16-Jul-2023_1234

#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 8 --ex_size 3 --layer -1 --n_epochs 1 --weight_decay 0.005 --lr 0.001 --grad_clipping 1 --minerva_dim 512 --skip_wandb --debug --num_minervas 3
#CUDA_VISIBLE_DEVICES=0 python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model wordLSTM --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --n_epochs 1 --weight_decay 0.005 --lr 0.001 --grad_clipping 1 --minerva_dim 512 --skip_wandb --debug

CUDA_VISIBLE_DEVICES=0 python VAL_ex_system_com.py --feats WhisperFull --model ExLSTM_layers --N 1 --exp_id test --batch_size 8 --ex_size 8 --layer -1 --num_validations 20 --minerva_dim 512 --pretrained_model_dir test_1_WhisperFull_ExLSTM_layers_14-29-15-Jul-2023_1234 --debug
