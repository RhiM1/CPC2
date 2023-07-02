
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
# import speechbrain as sb
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
from disjoint_val import get_disjoint_val_set
from process_cpc2_data import get_cpc2_dataset, prepare_dataset
from transformers import WhisperConfig, WhisperModel, \
    WhisperTokenizer, WhisperProcessor, Seq2SeqTrainer, \
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, pipeline
from datasets import Dataset

from models.ni_predictor_models import MetricPredictorLSTM, MetricPredictorAttenPool
from models.ni_feat_extractors import WhisperEncoder_feats
from models.ni_predictor_exemplar_models import ExemplarMetricPredictor

from constants import DATAROOT, DATAROOT_CPC1


def rmse_score(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def std_err(x, y):

    std = torch.std(x - y) / len(x)**0.5
    return std.item()


# def get_mean(scores):
#     out_list = []
#     for el in scores:
#         el = el.strip("[").strip("]").split(",")
#         el = [float(a) for a in el]
#         #print(el,type(el))
#         out_list.append(el)
#     return torch.Tensor(out_list).unsqueeze(0)


def get_whisper_feats(batch, args, model):

    input_features = torch.tensor(batch["input_features"]).to(args.device)
    # attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(args.device)
    # dat = {
    #     "input_features": input_features,
    #     # "attention_mask": attention_mask   
    # }
    model_out = model(input_features)
    # print(model_out)
    # print(model_out[0].size())
    # print(model_out.size())
    batch["feats"] = model_out

    return batch


def remove_dataset_columns(dataset, columns_to_keep):
    
    columns = [column for column in dataset.column_names if column not in columns_to_keep]
    # print(columns)
    dataset = dataset.remove_columns(columns)
    
    return dataset


def data_collator_fn(processor):
    def data_collator(features):        # split inputs and labels since they have to be of different lengths and need different padding methods

        input_features = [torch.tensor(feature["feats"]) for feature in features]
        # print(input_features)
        input_features = torch.stack(input_features)
        # attention_mask = [torch.tensor(feature["attention_mask"]) for feature in features]
        # print(attention_mask)
        # attention_mask = torch.stack(attention_mask)

        # # get the tokenized label sequences
        # label_features = [{"input_ids": feature["labels"]} for feature in features]
        # # pad the labels to max length
        # labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        # # replace padding with -100 to ignore loss correctly
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # # if bos token is appended in previous tokenization step,
        # # cut bos token here as it's append later anyways
        # if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        #     labels = labels[:, 1:]

        batch = {
            "input_features": input_features.to(args.device),
            # "attention_mask": attention_mask.to(args.device),
            # "labels": labels.to(args.device),
            "correctness": torch.tensor([feature['correctness'] / 100 for feature in features]).to(args.device)
        }

        return batch

    return data_collator


def validate_model(model, processor, data, optimizer, criterion, args):

    out_list = []
    model.eval()
    running_loss = 0.0
    loss_list = []

    data = remove_dataset_columns(data, columns_to_keep = ["feats", "correctness"])
    # data = data.set_format('torch')

    prepare_dataset_fn = data_collator_fn(processor)
    my_dataloader = DataLoader(dataset = data, batch_size = args.batch_size, collate_fn = prepare_dataset_fn)
    # my_dataloader = DataLoader(dataset = data, batch_size = args.batch_size, collate_fn = prepare_dataset_fn)

    print(f"batch_size: {args.batch_size}")
    print("Validating...")

    for batch in tqdm(my_dataloader, total = len(my_dataloader)):

        output, _ = model(batch["input_features"], packed_sequence = False)
        output = output[:, 0]
        loss = criterion(output, batch["correctness"])
        loss_list.append(loss.item())
        running_loss += loss.item()

        # print(output)
        # print(output.size())

        out_list.extend((output * 100).detach().cpu().tolist())
        # out_list.append(output.detach().cpu().numpy()[0][0]*100)

    return out_list, sum(loss_list) / len(loss_list)


def train_model(model, processor, data, optimizer, criterion, args):

    model.train()
        
    running_loss = 0.0
    loss_list = []

    
    data = remove_dataset_columns(data, columns_to_keep = ["feats", "correctness"])

    prepare_dataset_fn = data_collator_fn(processor)
    my_dataloader = DataLoader(dataset = data, batch_size = args.batch_size, collate_fn = prepare_dataset_fn)
    # my_dataloader = DataLoader(dataset = data, batch_size = args.batch_size)
    print(f"batch_size: {args.batch_size}")
    print("starting training...")
    
    for batch in tqdm(my_dataloader, total=len(my_dataloader)):

    
        optimizer.zero_grad()
        output, _ = model(batch["input_features"], packed_sequence = False)
        # print(output.size(), batch["correctness"].size())
        # print(output[:, 0].size(), batch["correctness"].size())
        loss = criterion(output[:, 0], batch["correctness"])
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        running_loss += loss.item()
        
    
    return model, optimizer, criterion, sum(loss_list) / len(loss_list)




def save_model(model,opt,epoch,args,val_loss):
    p = args.model_dir
    if not os.path.exists(p):
        os.mkdir(p)
    m_name = "%s-%s"%(args.model,args.seed)
    torch.save(model.state_dict(),"%s/%s_%s_%s_model.pt"%(p,m_name,epoch,val_loss))
    torch.save(opt.state_dict(),"%s/%s_%s_%s_opt.pt"%(p,m_name,epoch,val_loss))



def get_dis_val_set_losses(dis_val_preds, correct, validation, criterion, include_stats = False):
    
    # 1: disjoint validation set (listener) note: actually [1, 3, 5, 7]
    # 2: disjoint validation set (system) note: actually [2, 3, 6, 7]
    # 3: disjoint validation set (listener, system) note: actually [3, 7]
    # 4: disjoint validation set (scene) note: actually [4, 5, 6, 7]
    # 5: disjoint validation set (listener, scene) note: actually [5, 7]
    # 6: disjoint validation set (system, scene) note: actually [6, 7]
    # 7: disjoint validation set (listener, system, scene)

    # print(dis_val_preds)
    # print(correct)
    # print(validation)

    dis_val = {}
    dis_lis_val = {}
    dis_sys_val = {}
    dis_scene_val = {}

    
    dis_val_preds = torch.tensor(dis_val_preds)
    correct = torch.tensor(correct)
    validation = torch.tensor(validation)

    dis_val_bool = torch.zeros(len(dis_val_preds), dtype = torch.bool)
    dis_val_bool[validation == 7] = True
    dis_lis_val_bool = torch.zeros(len(dis_val_preds), dtype = torch.bool)
    dis_lis_val_bool[torch.isin(validation, torch.tensor([1, 3, 5, 7]))] = True
    dis_sys_val_bool = torch.zeros(len(dis_val_preds), dtype = torch.bool)
    dis_sys_val_bool[torch.isin(validation, torch.tensor([2, 3, 6, 7]))] = True
    dis_scene_val_bool = torch.zeros(len(dis_val_preds), dtype = torch.bool)
    dis_scene_val_bool[torch.isin(validation, torch.tensor([4, 5, 6, 7]))] = True

    if include_stats:
        dis_val_stats = get_stats(dis_val_preds[dis_val_bool].numpy() / 100, correct[dis_val_bool].numpy() / 100)
        dis_lis_val_stats = get_stats(dis_val_preds[dis_lis_val_bool].numpy() / 100, correct[dis_lis_val_bool].numpy() / 100)
        dis_sys_val_stats = get_stats(dis_val_preds[dis_sys_val_bool].numpy() / 100, correct[dis_sys_val_bool].numpy() / 100)
        dis_scene_val_stats = get_stats(dis_val_preds[dis_scene_val_bool].numpy() / 100, correct[dis_scene_val_bool].numpy() / 100)
        dis_val.update(dis_val_stats)
        dis_lis_val.update(dis_lis_val_stats)
        dis_sys_val.update(dis_sys_val_stats)
        dis_scene_val.update(dis_scene_val_stats)


    dis_val["loss"] = criterion(dis_val_preds[dis_val_bool] / 100, correct[dis_val_bool] / 100).item()
    dis_lis_val["loss"] = criterion(dis_val_preds[dis_lis_val_bool] / 100, correct[dis_lis_val_bool] / 100).item()
    dis_sys_val["loss"] = criterion(dis_val_preds[dis_sys_val_bool], correct[dis_sys_val_bool]).item()
    dis_scene_val["loss"] = criterion(dis_val_preds[dis_scene_val_bool] / 100, correct[dis_scene_val_bool] / 100).item()

    return dis_val, dis_lis_val, dis_sys_val, dis_scene_val


def get_stats(predictions, correctness):

    # print(f"predictions: {predictions}")
    # print(f"correctness: {correctness}")

    p_corr = np.corrcoef(predictions,correctness)[0][1]
    s_corr = spearmanr(predictions,correctness)[0]
    std = std_err(torch.tensor(predictions), torch.tensor(correctness))

    stats_dict = {
        "p_corr": p_corr,
        "s_corr": s_corr,
        "std": std
    }

    return stats_dict


def main(args, config):
    #set up the torch objects
    print("creating model: %s"%args.feats)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # feat_extractor = WhisperEncoder_feats(pretrained_model=args.pretrained_feats_model, use_cmvn = args.use_cmvn)
    feat_extractor = WhisperEncoder_feats(pretrained_model=args.pretrained_feats_model)
    dim_extractor = 768
    hidden_size = 768//2
    activation = nn.LeakyReLU
    att_pool_dim = 768

    if args.model == "LSTM":
        model = MetricPredictorLSTM(dim_extractor, hidden_size, activation, att_pool_dim)
    elif args.model == "AttenPool":
        model = MetricPredictorAttenPool(att_pool_dim)
    else:
        print("Model not recognised")
        exit(1)

    feat_extractor.eval()
    feat_extractor.requires_grad_(False)
    today = datetime.datetime.today()
    date = today.strftime("%H-%M-%d-%b-%Y")
    ex = "ex" if args.exemplar else ""

    model_name = "%s_%s_%s_%s_%s_%s_%s"%(args.exp_id,args.N,args.feats,args.model,ex,date,args.seed)
    model_dir = "save/%s"%(model_name)
    if not args.skip_wandb:
        # wandb_name = "%s_%s_%s_%s_feats_%s_%s"%(args.exp_id,args.N,args.model,ex,date,args.seed)
        run = wandb.init(
            project=args.wandb_project, 
            reinit = True, 
            name = model_name,
            tags = [f"N{args.N}", f"lr{args.lr}", args.feats, args.model, f"bs{args.batch_size}"]
        )
        if args.exemplar:
            run.tags = run.tags + ("exemplar")
    
    args.model_dir = model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    with open(model_dir + "/config.json", 'w+') as f:
        f.write(json.dumps(dict(config)))
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    if not args.use_CPC1:
        if int(args.in_json_file.split("/")[-1].split(".")[-2]) != int(args.N):
            print("Warning: N does not match dataset:")
            print(args.in_json_file.split("/")[-1].split(".")[-2],args.N)
            exit()

    whisper_processor = WhisperProcessor.from_pretrained(
        args.whisper_model, 
        language = args.whisper_language, 
        task = args.whisper_task
    )
    # whisper_processor.feature_extractor.chunk_length = 40
    # whisper_processor.feature_extractor.n_samples = whisper_processor.feature_extractor.chunk_length * whisper_processor.feature_extractor.sampling_rate
    # whisper_processor.feature_extractor.whisper_processor = whisper_processor.feature_extractor.n_samples // whisper_processor.feature_extractor.hop_length

    data = get_cpc2_dataset(args, whisper_processor, debug = False)

    # data['test'] = Dataset.from_dict(data['test'][0:10])
    # data['train'] = Dataset.from_dict(data['train'][0:10])
    # data['dis_val'] = Dataset.from_dict(data['dis_val'][0:10])


    feat_extractor = feat_extractor.to(args.device)
    fnKwargs = {"model": feat_extractor, "args": args}
    print("Extracting features...")
    data = data.map(get_whisper_feats, num_proc = 1, fn_kwargs = fnKwargs, batched = True, batch_size = 30)
    feat_extractor = feat_extractor.to('cpu')

    # print("Trainset: %s\nValset: %s\nDisValset: %s\nDisLisValset: %s\nDisSysValset: %s\nDisSceneValset: %s\nTestset: %s "%(train_data.shape[0],val_data.shape[0],dis_val_data.shape[0],dis_lis_val_data.shape[0],dis_sys_val_data.shape[0],dis_scene_val_data.shape[0],test_data.shape[0]))
    print("Trainset: %s\nValset: %s\nDisValset: %s"%(len(data["train"]), len(data["test"]), len(data["dis_val"])))
    print("=====================================")

    model = model.to(args.device)

    if args.do_train:
        print("Starting training of model: %s\nlearning rate: %s\nseed: %s\nepochs: %s\nsave location: %s/"%(args.model,args.lr,args.seed,args.n_epochs,args.model_dir))
        print("=====================================")
        for epoch in range(args.n_epochs):


            model,optimizer,criterion,training_loss = train_model(model, whisper_processor, data["train"], optimizer, criterion, args)

            _,val_loss = validate_model(model, whisper_processor, data["test"], optimizer, criterion, args)
            preds, all_dis_val_loss = validate_model(model, whisper_processor, data["dis_val"], optimizer, criterion, args)

            dis_val, dis_lis_val, dis_sys_val, dis_scene_val = get_dis_val_set_losses(
                dis_val_preds = preds,
                correct = data["dis_val"]["correctness"],
                validation = data["dis_val"]["validation"],
                criterion = criterion
            )

            # if args.test_json_file is not None:
            #     _,eval_loss = validate_model(model,test_data,optimizer,criterion,args)

            if not args.skip_wandb:
                log_dict = {
                    "val_rmse": val_loss**0.5,
                    "dis_val_rmse": dis_val["loss"]**0.5,
                    "dis_lis_val_rmse": dis_lis_val["loss"]**0.5,
                    "dis_sys_val_rmse": dis_sys_val["loss"]**0.5,
                    "dis_scene_val_rmse": dis_scene_val["loss"]**0.5,
                    "train_rmse": training_loss**0.5
                }
                # if args.test_json_file is not None:
                #     log_dict["eval_rmse"] = eval_loss**0.5
                
                wandb.log(log_dict)
            
            save_model(model,optimizer,epoch,args,val_loss)
            torch.cuda.empty_cache()
            print("Epoch: %s"%(epoch))
            print("\tTraining Loss: %s"%(training_loss**0.5))
            print("\tValidation Loss: %s"%(val_loss**0.5))
            print("\tDisjoint validation Loss: %s"%(dis_val["loss"]**0.5))
            print("\tDisjoint listener validation Loss: %s"%(dis_lis_val["loss"]**0.5))
            print("\tDisjoint system validation Loss: %s"%(dis_sys_val["loss"]**0.5))
            print("\tDisjoint scene validation Loss: %s"%(dis_scene_val["loss"]**0.5))
            # print("\tAll disjoint validation Loss: %s"%(all_dis_val_loss**0.5))
            print("=====================================")    
    print(model_dir)
    model_files = os.listdir(model_dir)

    model_files = [x for x in model_files if "model" in x]
    #print(model_files)
    model_files.sort(key=lambda x: float(x.split("_")[-2].strip(".pt")))
    model_file = model_files[0]
    print("Loading model:\n%s"%model_file)
    model.load_state_dict(torch.load("%s/%s"%(model_dir,model_file)))

    columns_to_keep = ["scene", "listener", "system", "correctness", "predicted"]
    
    # get validation predictions
    val_predictions, val_loss = validate_model(model, whisper_processor, data["test"],optimizer,criterion,args)
    val_error = val_loss**0.5
    val_stats = get_stats(val_predictions, data["test"]["correctness"])
    data["test"] = data["test"].add_column("predicted", val_predictions)
    removed_cols = remove_dataset_columns(dataset = data["test"], columns_to_keep = columns_to_keep)
    removed_cols.to_csv(args.out_csv_file + "_val_preds.csv")

    # get disjoint val predictions
    dis_val_predictions, dis_val_loss = validate_model(model,whisper_processor, data["dis_val"],optimizer,criterion,args)
    data["dis_val"] = data["dis_val"].add_column("predicted", dis_val_predictions)
    removed_cols = remove_dataset_columns(
        dataset = data["dis_val"],
        columns_to_keep = columns_to_keep
    )
    removed_cols.to_csv(args.out_csv_file + "_dis_val_preds.csv")

    dis_val, dis_lis_val, dis_sys_val, dis_scene_val = get_dis_val_set_losses(
        dis_val_preds = preds,
        correct = data["dis_val"]["correctness"],
        validation = data["dis_val"]["validation"],
        criterion = criterion,
        include_stats = True
    )
    
    #normalise the predictions
    val_gt = np.asarray(data["dis_val"]["correctness"])/100
    val_predictions = np.asarray(val_predictions)/100

    def logit_func(x,a,b):
     return 1/(1+np.exp(a*x+b))

    # logistic mapping curve fit to get the a and b parameters
    popt,_ = curve_fit(logit_func, val_predictions, val_gt)
    a,b = popt
    print("a: %s b: %s"%(a,b))
    print("=====================================")
    # Test the model
    # if args.test_json_file is not None:
    #     print("Testing model on test set")
    #     predictions,test_loss = validate_model(model,test_data,optimizer,criterion,args)
    #     predictions_fitted = np.asarray(predictions)/100
    #     #apply the logistic mapping
    #     predictions_fitted = logit_func(predictions_fitted,a,b)
    #     test_data["predicted"] = predictions
    #     test_data["predicted_fitted"] = predictions_fitted*100
    #     test_data[["scene", "listener", "system", "predicted","predicted_fitted"]].to_csv(args.out_csv_file + "_test_preds.csv", index=False)
    #     error = test_loss**0.5 * 100
    #     p_corr = np.corrcoef(np.array(predictions),test_data["correctness"])[0][1]
    #     s_corr = spearmanr(np.array(predictions),test_data["correctness"])[0]
    #     std = std_err(np.array(predictions), test_data["correctness"])
    #     error_fitted = rmse_score(np.array(predictions_fitted)*100,test_data["correctness"])
    #     p_corr_fitted = np.corrcoef(np.array(predictions_fitted)*100,test_data["correctness"])[0][1]
    #     s_corr_fitted = spearmanr(np.array(predictions_fitted)*100,test_data["correctness"])[0]
    #     std_fitted = std_err(np.array(predictions_fitted)*100,test_data["correctness"])

    #     print("Test Loss: %s"%test_loss)
    # else:
    #     print("Testing model on train+val set")
    #     predictions,test_loss = validate_model(model,data,optimizer,criterion,args)
    #     data["predicted"] = predictions
    #     data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file, index=False)
    
    with open (args.summ_file, "a") as f:
        csv_writer = csv.writer(f)
        # if args.test_json_file is not None:
        #     csv_writer.writerow([args.out_csv_file.split("/")[-1].strip(".csv"), val_error, val_p_corr, val_s_corr, val_std, error, std, p_corr, s_corr, error_fitted, std_fitted, p_corr_fitted, s_corr_fitted])
        # else:
        csv_writer.writerow([
            args.out_csv_file.split("/")[-1].strip(".csv"), 
            val_error, val_stats["p_corr"], val_stats["s_corr"], val_stats["std"], 
            dis_val["loss"]**0.5, dis_val["p_corr"], dis_val["s_corr"], dis_val["std"], 
            dis_lis_val["loss"]**0.5, dis_lis_val["p_corr"], dis_lis_val["s_corr"], dis_lis_val["std"], 
            dis_sys_val["loss"]**0.5, dis_sys_val["p_corr"], dis_sys_val["s_corr"], dis_sys_val["std"], 
            dis_scene_val["loss"]**0.5, dis_scene_val["p_corr"], dis_scene_val["s_corr"], dis_scene_val["std"]
        ])

    print("=====================================")
    
    if not args.skip_wandb:
        run.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # COnfiguration file
    parser.add_argument(
        "config_file", help="location of configuation json file", 
    )
    # data locations
    parser.add_argument(
        "--test_json_file", help="JSON file containing the CPC2 test metadata", default = None
    )
    parser.add_argument(
        "--processed_data_file", help="location of pre-processed datafile" , default=f"huggingface_data"
    )

    # Model
    parser.add_argument(
        "--model", help="model type" , default="999",
    )

    # Feat extraction
    parser.add_argument(
        "--feats", help="feats extractor" , default="999",
    )
    parser.add_argument(
        "--pretrained_feats_model", help="model type" , default=None,
    )
    parser.add_argument(
        "--use_cmvn", help="include cepstral mean and variance normalisation" , default=False, action='store_true'
    )
    parser.add_argument(
        "--whisper_model", help="location of configuation json file", default = "openai/whisper-small"
    )
    parser.add_argument(
        "--trained_model", help="location of configuation json file", default = "openai/whisper-small"
    )
    parser.add_argument(
        "--whisper_language", help="location of configuation json file", default = "English"
    )
    parser.add_argument(
        "--whisper_task", help="location of configuation json file", default = "transcribe"
    )


    # Training hyperparams
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=0, type=int
    )
    parser.add_argument(
        "--lr", help="learning rate", default=999, type=float
    )
    parser.add_argument(
        "--batch_size", help="batch size" , default=1, type=int
    )

    # General
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
    
    config["use_cmvn"] = args.use_cmvn

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
        args.in_json_file = f"{DATAROOT}metadata/CEC1.train.{args.N}.json"
        config["in_json_file"] = args.in_json_file
        
    if args.use_cmvn:
        args.processed_data_file = f"{DATAROOT}{args.processed_data_file}_N{args.N}_cmvn{int(args.use_cmvn)}"
    else:
        args.processed_data_file = f"{DATAROOT}{args.processed_data_file}_N{args.N}"
        
    if args.summ_file is None:
        if args.use_CPC1:
            args.summ_file = "save/CPC1_metrics.csv"
        else:
            args.summ_file = "save/CPC2_metrics.csv"

    args.out_csv_file = f"save/{args.exp_id}_N{args.N}_{args.feats}_{args.model}"
    config["out_csv_file"] = args.out_csv_file
    
    config["device"] = args.device

    main(args, config)
