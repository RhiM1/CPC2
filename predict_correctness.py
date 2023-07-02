import torch
import argparse
import numpy as np
import os
from attrdict import AttrDict
import wandb
import json


from constants import DATAROOT, DATAROOT_CPC1




def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", help="location of configuation json file", 
    )
    parser.add_argument(
        "--test_json_file", help="JSON file containing the CPC2 test metadata", 
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=0, type=int
    )
    parser.add_argument(
        "--lr", help="learning rate", default=999, type=float
    )
    parser.add_argument(
        "--feats", help="feats extractor" , default="999",
    )
    parser.add_argument(
        "--model", help="model type" , default="999",
    )
    parser.add_argument(
        "--seed", help="torch seed" , default=999,
    )
    parser.add_argument(
        "--do_train", help="do training", default=True, type=bool
    )
    parser.add_argument(
        "--skip_wandb", help="skip logging via WandB", default=False, action='store_true'
    )
    parser.add_argument(
        "--exp_id", help="id for individual experiment"
    )
    parser.add_argument(
        "--batch_size", help="batch size" , default=999, type=int
    )
    parser.add_argument(
        "--use_CPC1", help="train and evaluate on CPC1 data" , default=False, action='store_true'
    )
    parser.add_argument(
        "--summ_file", help="train and evaluate on CPC1 data" , default=None
    )
    parser.add_argument(
        "--N", help="train split" , default=1, type=int
    )
    parser.add_argument(
        "--wandb_project", help="W and B project name" , default=None
    )

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_filename = "{}.json".format(args.config_file)
    with open(os.path.join("configs", config_filename)) as f:
        config = AttrDict(json.load(f))

    config["N"] = args.N
    args.exemplar = config["exemplar"]

    if args.test_json_file is None:
        args.test_json_file = config["test_json_file"]
    else:
        config["test_json_file"] = args.test_json_file

    if args.n_epochs == 0:
        args.n_epochs = config["n_epochs"]
    else:
        config["n_epochs"] = args.n_epochs

    if args.lr == 999:
        args.lr = config["lr"]
    else:
        config["lr"] = args.lr

    if args.feats == "999":
        args.feats = config["feats"]
    else:
        config["feats"] = args.feats

    if args.model == "999":
        args.model = config["model"]
    else:
        config["model"] = args.model

    if args.seed == 999:
        args.seed = config["seed"]
    else:
        config["seed"] = args.seed

    if not args.do_train:
        config["do_train"] = False

    if args.skip_wandb:
        config["skip_wandb"] = True

    if args.exp_id is not None:
        config["exp_id"] = args.exp_id
    else:
        args.exp_id = config["exp_id"]

    if args.batch_size == 999:
        args.batch_size = config["batch_size"]
    else:
        config["batch_size"] = args.batch_size

    if args.use_CPC1:
        args.wandb_project = "CPC1"
        config["wandb_project"] = "CPC1"
        args.dataroot = DATAROOT_CPC1
        args.in_json_file = DATAROOT_CPC1 + "metadata/CPC1.train_indep.json"
        config["in_json_file"] = DATAROOT_CPC1 + "metadata/CPC1.train_indep.json"
        args.test_json_file = DATAROOT_CPC1 + "metadata/CPC1.test_indep.json"
        config["test_json_file"] = DATAROOT_CPC1 + "metadata/CPC1.test_indep.json"
    else:
        args.dataroot = DATAROOT
        args.wandb_project = "CPC2" if args.wandb_project is None else args.wandb_project
        config["wandb_project"] = args.wandb_project
        args.in_json_file = f"{DATAROOT}clarity_data/metadata/CEC1.train.{args.N}.json"
        config["in_json_file"] = args.in_json_file
        
    if args.summ_file is None:
        if args.use_CPC1:
            args.summ_file = "save/CPC1_metrics.csv"
        else:
            args.summ_file = "save/CPC2_metrics.csv"

    args.out_csv_file = f"save/{args.exp_id}_N{args.N}_{args.feats}_{args.model}"
    config["out_csv_file"] = args.out_csv_file
    
    config["device"] = args.device

    main(args, config)
