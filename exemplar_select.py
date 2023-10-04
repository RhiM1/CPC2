import argparse
#from GHA import audiogram

import numpy as np
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import speechbrain as sb
from torch.utils.data import DataLoader
import os
import datetime
import torchaudio.transforms as T
from scipy.optimize import curve_fit
import wandb
from attrdict import AttrDict
import json
import csv
from scipy.stats import spearmanr
from data_handling import get_disjoint_val_set
from process_cpc2_data import get_cpc2_dataset
from transformers import WhisperProcessor
import random
from models.ni_predictor_models import MetricPredictorLSTM, MetricPredictorAttenPool, ExLSTM, \
    MetricPredictorLSTM_layers, ExLSTM_layers, ExLSTM_std, ExLSTM_log, ExLSTM_div, wordLSTM
from models.ni_feat_extractors import Spec_feats, XLSREncoder_feats, XLSRFull_feats, \
    HuBERTEncoder_feats, HuBERTFull_feats, WhisperEncoder_feats, WhisperFull_feats, WhisperBase_feats
from exemplar import get_ex_set

from constants import DATAROOT, DATAROOT_CPC1


def get_listener_ex_set(args):

    pass


def main(args):

    pass




if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", help="random seed for repeatability", default=1234, type=int
    )
    parser.add_argument(
        "--debug", help="use a tiny dataset for debugging", default=False, action='store_true'
    )
    parser.add_argument(
        "--save_feats", help="save extracted feats to disk for future use", default=False, action='store_true'
    )
    parser.add_argument(
        "--extract_feats", help="extract feats rather than loading from file", default=False, action='store_true'
    )
    parser.add_argument(
        "--pretrained_model_dir", help="directory of pretrained model", default=None
    )


    parser.add_argument(
        "--N", help="train split" , default=1, type=int
    )
    parser.add_argument(
        "--train_disjoint", help="include the disjoin val data in training data (for final model only)", default=False, action='store_true'
    )

    # Feats details
    parser.add_argument(
        "--feats_model", help="feats extractor model" , default=None,
    )
    parser.add_argument(
        "--layer", help="layer of feat extractor to use" , default=None, type = int
    )
    parser.add_argument(
        "--num_layers", help="layer of feat extractor to use" , default=12, type = int
    )
    parser.add_argument(
        "--whisper_model", help="location of configuation json file", default = "openai/whisper-small"
    )
    parser.add_argument(
        "--whisper_language", help="location of configuation json file", default = "English"
    )
    parser.add_argument(
        "--whisper_task", help="location of configuation json file", default = "transcribe"
    )

    # # Classifier model 
    # parser.add_argument(
    #     "--model", help="model type" , default=None,
    # )
    # parser.add_argument(
    #     "--restart_model", help="path to a previous training checkpoint" , default=None,
    # )
    # parser.add_argument(
    #     "--restart_epoch", help="epoch number to restart from (for logging)" , default=0, type = int
    # )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", help="batch size" , default=None, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=None, type=int
    )
    parser.add_argument(
        "--lr", help="learning rate", default=None, type=float
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", default=None, type=float
    )
    parser.add_argument(
        "--grad_clipping", help="gradient clipping", default=1, type=float
    )

    # Exemplar bits
    parser.add_argument(
        "--ex_size", help="train split" , default=None, type=int
    )
    parser.add_argument(
        "--p_factor", help="exemplar model p_factor" , default=None, type=int
    )
    parser.add_argument(
        "--minerva_dim", help="exemplar model feat transformation dimnesion" , default=None, type=int
    )
    parser.add_argument(
        "--minerva_r_dim", help="exemplar model feat transformation dimnesion" , default=None, type=int
    )
    parser.add_argument(
        "--num_minervas", help="number of minerva models to use for system combinations or the std model" , default=None, type=int
    )
    parser.add_argument(
        "--random_exemplars", help="use randomly selected exemplars, rather than stratified exemplars", default=False, action='store_true'
    )
    parser.add_argument(
        "--exemplar_source", help="one of CEC1, CEC2, all or matched", default="CEC2"
    )
    parser.add_argument(
        "--use_r_encoding", help="use positional encoding for exemplar correctness", default=False, action='store_true'
    )

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)