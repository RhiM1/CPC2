
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
    MetricPredictorLSTM_layers, ExLSTM_layers, ExLSTM_std
from models.ni_feat_extractors import Spec_feats, XLSREncoder_feats, XLSRFull_feats, \
    HuBERTEncoder_feats, HuBERTFull_feats, WhisperEncoder_feats, WhisperFull_feats, WhisperBase_feats
from exemplar import get_ex_set

from constants import DATAROOT, DATAROOT_CPC1


def main(args, config):
    #set up the torch objects
    print("creating model: %s"%args.feats_model)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.exemplar = False

    today = datetime.datetime.today()
    date = today.strftime("%H-%M-%d-%b-%Y")

    model_name = "%s_%s_%s_%s_%s_%s"%(args.exp_id,args.N,args.feats_model,args.model,date,args.seed)
    model_dir = "save/%s"%(model_name)


    # Make a model directory (if it doesn't exist) and write out config for future reference
    args.model_dir = model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    with open(model_dir + "/config.json", 'w+') as f:
        f.write(json.dumps(dict(config)))
        
    criterion = nn.MSELoss()
    
    data = pd.read_json(args.in_json_file)
    data["subset"] = "CEC1"
    if not args.use_CPC1:
        data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
        data2["subset"] = "CEC2"
        data = pd.concat([data, data2])
    data = data.drop_duplicates(subset = ['signal'], keep = 'last')
    data["predicted"] = np.nan  

    if args.debug:
        _, data = train_test_split(data, test_size = 0.01)

    # Reset seeds for repeatable behaviour regardless of
    # how feats / models are obtained
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Mark various disjoint validation sets in the data
    data = get_disjoint_val_set(args, data)

    # Split into train and val sets
    dis_val_data = data[data.validation > 0].copy()
    train_data,val_data = train_test_split(data[data.validation == 0],test_size=0.1)

    train_data[["scene", "listener", "system", "subset", "correctness"]].to_csv(args.out_csv_file + "_train.csv", index=False)
    val_data[["scene", "listener", "system", "subset", "correctness"]].to_csv(args.out_csv_file + "_val.csv", index=False)
    dis_val_data[["scene", "listener", "system", "subset", "validation", "correctness"]].to_csv(args.out_csv_file + "_all_dis.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation == 7].copy()
    temp_data[["scene", "listener", "system", "subset", "correctness"]].to_csv(args.out_csv_file + "_dis.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation.isin([1, 3, 5, 7])].copy()
    temp_data[["scene", "listener", "system", "subset", "correctness"]].to_csv(args.out_csv_file + "_dis_lis.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation.isin([2, 3, 6, 7])].copy()
    temp_data[["scene", "listener", "system", "subset", "correctness"]].to_csv(args.out_csv_file + "_dis_sys.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation.isin([4, 5, 6, 7])].copy()
    temp_data[["scene", "listener", "system", "subset", "correctness"]].to_csv(args.out_csv_file + "_dis_scene.csv", index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", help="location of configuation json file", default = None #"configs/default_config.json"
    )
    parser.add_argument(
        "--exp_id", help="id for individual experiment", default = None
    )
    parser.add_argument(
        "--summ_file", help="path to write summary results to" , default="save/CPC2_metrics.csv"
    )
    parser.add_argument(
        "--out_csv_file", help="path to write the preditions to" , default=None
    )
    parser.add_argument(
        "--wandb_project", help="W and B project name" , default="CPC2"
    )
    parser.add_argument(
        "--skip_wandb", help="skip logging via WandB", default=False, action='store_true'
    )
    parser.add_argument(
        "--seed", help="random seed for repeatability", default=1234,
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


    # Training data
    parser.add_argument(
        "--do_train", help="do training", default=True, type=bool
    )
    parser.add_argument(
        "--use_CPC1", help="train and evaluate on CPC1 data" , default=False, action='store_true'
    )
    parser.add_argument(
        "--N", help="train split" , default=1, type=int
    )

    # Test data
    parser.add_argument(
        "--do_eval", help="get predictions for the evaluation set", default=False, action='store_true'
    )

    # Feats extractor
    #   -WhisperEncoder
    #   -WhisperFull
    #   - ...
    parser.add_argument(
        "--feats_model", help="feats extractor model" , default=None,
    )
    parser.add_argument(
        "--pretrained_feats_model", help="the name of the pretrained model to use - can be a local save" , default=None,
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

    # Classifier model 
    parser.add_argument(
        "--model", help="model type" , default=None,
    )
    parser.add_argument(
        "--restart_model", help="path to a previous training checkpoint" , default=None,
    )
    parser.add_argument(
        "--restart_epoch", help="epoch number to restart from (for logging)" , default=0, type = int
    )

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
        "--grad_clipping", help="gradient clipping", default=None, type=float
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
        "--num_minervas", help="number of minerva models to use for system combinations or the std model" , default=None, type=int
    )
    parser.add_argument(
        "--random_exemplars", help="use randomly selected exemplars, rather than stratified exemplars", default=False, action='store_true'
    )
    parser.add_argument(
        "--matching_exemplars", help="use CEC 1 & 2 exemplars for training and CEC 2 exemplars for vliadation", default=False, action='store_true'
    )




    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            config = AttrDict(json.load(f))
    else:
        config = AttrDict()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config["device"] = args.device

    if args.exp_id is None:
        args.exp_id = config["exp_id"] if "exp_id" in config else "test"
    config["exp_id"] = args.exp_id

    if args.wandb_project is None:
        args.wandb_project = config["wandb_project"] if "wandb_project" in config else None
    config["wandb_project"] = args.wandb_project
    
    if args.seed is None:
        args.seed = config["seed"] if "seed" in config else 1234
    config["seed"] = args.seed

    config["debug"] = args.debug
    config["do_train"] = args.do_train
    config["use_CPC1"] = args.use_CPC1
    config["N"] = args.N
    config["do_eval"] = args.do_eval

    if args.feats_model is None:
        args.feats_model = config["feats_model"] if "feats_model" in config else "WhisperFull"
    config["feats_model"] = args.feats_model

    if args.pretrained_feats_model is None:
        args.pretrained_feats_model = config["pretrained_feats_model"] if "pretrained_feats_model" in config else None
    config["pretrained_feats_model"] = args.pretrained_feats_model

    if args.layer == None:
        args.layer = config["layer"] if "layer" in config else None
    config["layer"] = args.layer

    config["whisper_model"] = args.whisper_model
    config["whisper_language"] = args.whisper_language
    config["whisper_task"] = args.whisper_task

    if args.model is None:
        args.model = config["model"] if "model" in config else None
    config["model"] = args.model

    config["restart_model"] = args.restart_model
    config["restart_epoch"] = args.restart_epoch

    if args.batch_size is None:
        args.batch_size = config["batch_size"] if "batch_size" in config else 1
    config["batch_size"] = args.batch_size

    if args.lr is None:
        args.lr = config["lr"] if "lr" in config else 0.001
    config["lr"] = args.lr

    if args.weight_decay is None:
        args.weight_decay = config["weight_decay"] if "weight_decay" in config else 0
    config["weight_decay"] = args.weight_decay

    if args.n_epochs is None:
        args.n_epochs = config["n_epochs"] if "n_epochs" in config else 10
    config["n_epochs"] = args.n_epochs
    
    if args.ex_size is None:
        args.ex_size = config["ex_size"] if "ex_size" in config else None
    config["ex_size"] = args.ex_size

    if args.p_factor is None:
        args.p_factor = config["p_factor"] if "p_factor" in config else 1
    config["p_factor"] = args.p_factor

    if args.minerva_dim is None:
        args.minerva_dim = config["minerva_dim"] if "minerva_dim" in config else None
    config["minerva_dim"] = args.minerva_dim
    
    if args.num_minervas is None:
        args.num_minervas = config["num_minervas"] if "num_minervas" in config else 1
    config["num_minervas"] = args.num_minervas
    
    config["random_exemplars"] = args.random_exemplars
    config["matching_exemplars"] = args.matching_exemplars

    if args.use_CPC1:
        args.wandb_project = "CPC1" if args.wandb_project is None else args.wandb_project
        config["wandb_project"] = "CPC1"
        args.dataroot = DATAROOT_CPC1
        args.in_json_file = DATAROOT_CPC1 + "metadata/CPC1.train_indep.json"
        config["in_json_file"] = args.in_json_file
        args.test_json_file = DATAROOT_CPC1 + "metadata/CPC1.test_indep.json"
        config["test_json_file"] = args.test_json_file
    else:
        args.dataroot = DATAROOT
        args.wandb_project = "CPC2" if args.wandb_project is None else args.wandb_project
        config["wandb_project"] = args.wandb_project
        args.in_json_file = f"{DATAROOT}/metadata/CEC1.train.{args.N}.json"
        config["in_json_file"] = args.in_json_file
        if args.do_eval:
            args.test_json_file = f"{DATAROOT}/metadata/CEC2.test.{args.N}.json"
            config["test_json_file"] = args.test_json_file
        else:
            args.test_json_file = None
            config["test_json_file"] = None
        
    if args.summ_file is None:
        if args.use_CPC1:
            args.summ_file = "save/CPC1_metrics.csv"
        else:
            args.summ_file = "save/CPC2_metrics.csv"
    config["summ_file"] = args.summ_file

    if args.out_csv_file is None:
        args.out_csv_file = f"save/base_N{args.N}"
    config["out_csv_file"] = args.out_csv_file
    
    main(args, config)
