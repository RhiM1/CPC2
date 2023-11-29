
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
    MetricPredictorLSTM_layers, ExLSTM_layers, ExLSTM_std, ExLSTM_log, ExLSTM_div, wordLSTM,\
    ExLSTM_layers_learn
from models.ni_feat_extractors import Spec_feats, XLSREncoder_feats, XLSRFull_feats, \
    HuBERTEncoder_feats, HuBERTFull_feats, WhisperEncoder_feats, WhisperFull_feats, WhisperBase_feats
from exemplar import get_ex_set

from constants import DATAROOT, DATAROOT_CPC1

def rmse_score(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def std_err(x, y):
    return np.std(x - y) / np.sqrt(len(x))


def compute_feats(wavs,resampler):
    """Feature computation pipeline"""
    wavs_l = wavs[:,:,0]
    wavs_r = wavs[:,:,1]
    wavs_l = resampler(wavs_l)
    wavs_r = resampler(wavs_r)
    return wavs_l,wavs_r


def audio_pipeline(path,fs=32000):
    wavs = sb.dataio.dataio.read_audio_multichannel(path)    
    return wavs


def format_correctness(y):
    y = torch.tensor([y])
    y = y/100
    return y


def format_feats(y):
    y = torch.from_numpy(y).to(torch.float)
    return y[0]


def get_stats(predictions, correctness):
    p_corr = np.corrcoef(predictions,correctness)[0][1]
    s_corr = spearmanr(predictions,correctness)[0]
    std = std_err(predictions, correctness)
    stats_dict = {
        "p_corr": p_corr,
        "s_corr": s_corr,
        "std": std
    }
    return stats_dict


def get_dis_val_set_losses(dis_val_preds, correct, validation, criterion, include_stats = False):
    
    # Urgh, ugly
    # 1: disjoint validation set (listener) note: actually [1, 3, 5, 7]
    # 2: disjoint validation set (system) note: actually [2, 3, 6, 7]
    # 3: disjoint validation set (listener, system) note: actually [3, 7]
    # 4: disjoint validation set (scene) note: actually [4, 5, 6, 7]
    # 5: disjoint validation set (listener, scene) note: actually [5, 7]
    # 6: disjoint validation set (system, scene) note: actually [6, 7]
    # 7: disjoint validation set (listener, system, scene)

    stats = [0] * 4
    vals = [[7],        
        [1, 3, 5, 7], 
        [2, 3, 6, 7], 
        [4, 5, 6, 7]]
    
    dis_val_preds = torch.tensor(dis_val_preds)
    correct = torch.tensor(correct)
    validation = torch.tensor(validation)

    for i in range(len(stats)):
        stats[i] = {}
        dis_val_bool = torch.zeros(len(dis_val_preds), dtype = torch.bool)
        dis_val_bool[torch.isin(validation, torch.tensor(vals[i]))] = True

        if include_stats:
            dis_val_stats = get_stats(dis_val_preds[dis_val_bool].numpy() / 100, correct[dis_val_bool].numpy() / 100)
            stats[i].update(dis_val_stats)

        stats[i]["loss"] = criterion(dis_val_preds[dis_val_bool] / 100, correct[dis_val_bool] / 100).item()

    return stats[0], stats[1], stats[2], stats[3]


def extract_feats(feat_extractor, data, args, theset, save_feats_file = None):
    feat_extractor.eval()
    name_list = data["signal"]
    correctness_list = data["correctness"]
    listener_list = data["listener"]
    scene_list = data["scene"]
    subset_list = data["subset"]

    feat_extractor.eval()
    feat_extractor.to(args.device)

    test_dict = {}
    
    for sub,name,corr,scene,lis in zip(subset_list,name_list,correctness_list,scene_list,listener_list):
        test_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis}
    
    dynamic_items = [
        {"func": lambda l: format_correctness(l),
        "takes": "correctness",
        "provides": "formatted_correctness"},
        
    ]
    if args.use_CPC1:
        dynamic_items.append(
            {"func": lambda l: audio_pipeline("%s/HA_outputs/%s/%s.wav"%(args.dataroot,theset,l),32000),
            "takes": ["signal"],
            "provides": "wav"}
        )
    else:
        dynamic_items.append(
            {"func": lambda l,y: audio_pipeline("%s/HA_outputs/%s.%s/%s/%s.wav"%(args.dataroot,theset,args.N,y,l),32000),
            "takes": ["signal","subset"],
            "provides": "wav"}
        )

    data_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
    data_set.set_output_keys(["wav", "formatted_correctness"])

    feats_list_l = []
    feats_list_r = []
    correct_list = []
    
    resampler = T.Resample(32000, 16000).to(args.device)

    my_dataloader = DataLoader(data_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("Extracting features...")
    for batch in tqdm(my_dataloader):

        batch = batch.to(args.device)
        wavs, correctness = batch
        correctness = correctness.data

        wavs_data = wavs.data
        feats_l,feats_r = compute_feats(wavs_data,resampler)
       
        extracted_feats_l = feat_extractor(feats_l.float())
        extracted_feats_r = feat_extractor(feats_r.float())

        feats_list_l.append(extracted_feats_l.detach().cpu().numpy())
        feats_list_r.append(extracted_feats_r.detach().cpu().numpy())
        correct_list.append(correctness)

    data['feats_l'] = feats_list_l
    data['feats_r'] = feats_list_r

    # print(f"size 0 l: {feats_list_l[0]}")
    # print(f"size 0 r: {feats_list_r[0]}")
    print(f"size 0 l: {feats_list_l[0].shape}")
    print(f"size 0 r: {feats_list_r[0].shape}")
    # print(f"one layer size 0 l: {feats_list_l[0][:, :, :, 0].shape}")
    # print(f"one layer size 0 r: {feats_list_r[0][:, :, :, 0].shape}")


    if save_feats_file is not None:
        if args.layer == -1:
            for layer in range(args.num_layers):
                # print(f"{save_feats_file}{layer}_l.txt")
                with open(f"{save_feats_file}{layer}_l.txt", "w") as f:
                    f.write("\n".join(" ".join(map(str, feats[:, :, :, layer].flatten())) for feats in feats_list_l))
                with open(f"{save_feats_file}{layer}_r.txt", "w") as f:
                    f.write("\n".join(" ".join(map(str, feats[:, :, :, layer].flatten())) for feats in feats_list_r))

        else:
            print(f"{save_feats_file}_l.txt")
            with open(f"{save_feats_file}_l.txt", "w") as f:
                f.write("\n".join(" ".join(map(str, feats.flatten())) for feats in feats_list_l))
            with open(f"{save_feats_file}_r.txt", "w") as f:
                f.write("\n".join(" ".join(map(str, feats.flatten())) for feats in feats_list_r))

    return data


def get_dynamic_dataset(data):

    name_list = data["signal"]
    correctness_list = data["correctness"]
    scene_list = data["scene"]
    listener_list = data["listener"]
    subset_list = data["subset"]
    feats_l_list = data["feats_l"]
    feats_r_list = data["feats_r"]
   
    data_dict = {}
    for sub,name,corr,scene,lis, f_l, f_r in  zip(subset_list,name_list,correctness_list,scene_list,listener_list, feats_l_list, feats_r_list):
        data_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_l":f_l, "feats_r": f_r}
    

        dynamic_items = [
            {"func": lambda l: format_correctness(l),
            "takes": "correctness",
            "provides": "formatted_correctness"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_l",
            "provides": "formatted_feats_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_r",
            "provides": "formatted_feats_r"}
        ]

    ddata = sb.dataio.dataset.DynamicItemDataset(data_dict,dynamic_items)
    ddata.set_output_keys(["formatted_correctness", "formatted_feats_l", "formatted_feats_r"])

    return ddata


def validate_model(model,test_data,optimizer,criterion,args,ex_data = None):
    
    out_list = []
    model.eval()
    running_loss = 0.0
    loss_list = []

    test_set = get_dynamic_dataset(test_data)
    my_dataloader = DataLoader(test_set,args.batch_size,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = False)

    # Get exemplar data loader if using exemplars
    if ex_data is not None:
        if args.random_exemplars:
            len_ex = len(test_data)
            ex_set = get_dynamic_dataset(test_data)
            ex_dataloader = DataLoader(ex_set,args.ex_size*args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = True)
        else:
            ex_set = get_ex_set(ex_data, args)
            len_ex = len(ex_set)
            ex_set = get_dynamic_dataset(ex_set)
            ex_dataloader = DataLoader(ex_set,args.ex_size*args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = False)
        ex_used = 0
        ex_dataloader = iter(ex_dataloader)

    print("starting validation...")
    for batch in tqdm(my_dataloader):
       
        correctness, feats_l, feats_r = batch

        # Use packed sequences for batches
        feats_l = torch.nn.utils.rnn.pack_padded_sequence(
            feats_l.data, 
            (feats_l.lengths * feats_l.data.size(1)).to(torch.int64), 
            batch_first=True,
            enforce_sorted = False
        )
        feats_r = torch.nn.utils.rnn.pack_padded_sequence(
            feats_r.data, 
            (feats_r.lengths * feats_r.data.size(1)).to(torch.int64), 
            batch_first=True,
            enforce_sorted = False
        )
        feats_l = feats_l.to(args.device)
        feats_r = feats_r.to(args.device)

        # Get exemplar for this minibatch, if using exemplar model
        if ex_data is not None:
            ex_used += args.num_minervas * args.ex_size
            if ex_used > len_ex:
                if args.random_exemplars:
                    ex_dataloader = iter(DataLoader(ex_set,args.ex_size*args.num_minervas, collate_fn=sb.dataio.batch.PaddedBatch, shuffle = True))
                else:
                    ex_set = get_ex_set(ex_data, args)
                    ex_set = get_dynamic_dataset(ex_data)
                    ex_dataloader = iter(DataLoader(ex_set,args.ex_size*args.num_minervas, collate_fn=sb.dataio.batch.PaddedBatch, shuffle = False))
                ex_used = args.num_minervas * args.ex_size

            exemplars = next(ex_dataloader)
            ex_correct, ex_feats_l, ex_feats_r = exemplars
            # print(f"ex_correct:\n{ex_correct}")
            ex_feats_l = torch.nn.utils.rnn.pack_padded_sequence(
                ex_feats_l.data, 
                (ex_feats_l.lengths * ex_feats_l.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            ex_feats_r = torch.nn.utils.rnn.pack_padded_sequence(
                ex_feats_r.data, 
                (ex_feats_r.lengths * ex_feats_r.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            ex_feats_l = ex_feats_l.to(args.device)
            ex_feats_r = ex_feats_r.to(args.device)
            ex_correct = ex_correct.data.to(args.device)
            
        target_scores = correctness.data.to(args.device)

        if ex_data is not None:
            output_l, _ = model(feats_l, ex_feats_l, ex_correct, num_minervas = args.num_minervas)
            output_r, _ = model(feats_r, ex_feats_r, ex_correct, num_minervas = args.num_minervas)
        elif args.exemplar:
            output_l, _ = model(feats_l, left = True, num_minervas = args.num_minervas)
            output_r, _ = model(feats_r, left = False, num_minervas = args.num_minervas)
        else:
            output_l,_ = model(feats_l)
            output_r,_ = model(feats_r)

        if args.learn_incorrect:
            output = torch.minimum(output_l,output_r)
        else:
            output = torch.maximum(output_l,output_r)
        loss = criterion(output,target_scores)

        for out_val in output:
            # print(out_val)
            out_list.append(out_val.detach().cpu().numpy()[0]*100)

        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item() * len(output)
    return out_list, running_loss / len(test_data)


def get_feats(data, save_feats_file, dim_extractor, feat_extractor, theset, args):
    
    if args.extract_feats:
        files_exist = False
    elif args.layer == -1:
        files_exist = True
        for layer in range(args.num_layers):
            files_exist = files_exist if os.path.exists(f"{save_feats_file}{layer}_l.txt") else False
            files_exist = files_exist if os.path.exists(f"{save_feats_file}{layer}_r.txt") else False
    else:
        files_exist = os.path.exists(save_feats_file + "_l.txt") and os.path.exists(save_feats_file + "_r.txt")
    
    # Load feats from file if they exist (only works for Whisper decoder)
    if args.layer != -1 and files_exist:
        print("Loading feats from file...")
        
        with open(save_feats_file + "_l.txt", "r") as f:
            feats_l = f.readlines()
        with open(save_feats_file + "_r.txt", "r") as f:
            feats_r = f.readlines()

        feats_l = [np.fromstring(feats, dtype=float, sep=' ') for feats in feats_l]
        feats_r = [np.fromstring(feats, dtype=float, sep=' ') for feats in feats_r]
        feats_l = [feats.reshape(1, -1, dim_extractor) for feats in feats_l]
        feats_r = [feats.reshape(1, -1, dim_extractor) for feats in feats_r]

        data['feats_l'] = feats_l
        data['feats_r'] = feats_r

    elif args.layer == -1 and files_exist:
        print("Loading feats from file...")
        
        feats_l = []
        feats_r = []
        for layer in range(args.num_layers):
            print(f"Loading layer {layer}...")
            with open(f"{save_feats_file}{layer}_l.txt", "r") as f:
                feats_l_layer = f.readlines()
            with open(f"{save_feats_file}{layer}_r.txt", "r") as f:
                feats_r_layer = f.readlines()
            feats_l.append([np.fromstring(feats, dtype=float, sep=' ') for feats in feats_l_layer])
            feats_r.append([np.fromstring(feats, dtype=float, sep=' ') for feats in feats_r_layer])
            feats_l[layer] = [feats.reshape(1, -1, dim_extractor) for feats in feats_l[layer]]
            feats_r[layer] = [feats.reshape(1, -1, dim_extractor) for feats in feats_r[layer]]

        feats_l_all = []
        feats_r_all = []
        for row in range(len(feats_l[0])):
            temp_feats_l = []
            temp_feats_r = []
            for layer in range(len(feats_l)):
                temp_feats_l.append(feats_l[layer][row])
                temp_feats_r.append(feats_r[layer][row])
            feats_l_all.append(np.stack(temp_feats_l, axis = -1))
            feats_r_all.append(np.stack(temp_feats_r, axis = -1))

        data['feats_l'] = feats_l_all
        data['feats_r'] = feats_r_all
        # feats_l = np.stack([feats_l[layer][row] for row in feats_l[layer] for layer in len(feats_l)])

    else:
        # Extract feats
        data = extract_feats(
            feat_extractor, 
            data, 
            args, 
            theset, 
            save_feats_file = save_feats_file if args.save_feats else None
        )
        feat_extractor.to('cpu')

    return data


def format_audiogram(audiogram):
    """
     "audiogram_cfs": [
      250,
      500,
      1000,
      2000,
      3000,
      4000,
      6000,
      8000
    ],
    
    TO
    [250, 500, 1000, 2000, 4000, 6000]
    """
    audiogram = numpy.delete(audiogram,4)
    audiogram = audiogram[:-1]
    return audiogram
    

def train_model(model,train_data,optimizer,criterion,args,ex_data=None):
    model.train()
        
    running_loss = 0.0
    loss_list = []

    train_set = get_dynamic_dataset(train_data)
    my_dataloader = DataLoader(train_set,args.batch_size,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = True)

    # Set up exemplar data loader if using exemplars
    if ex_data is not None:
        if args.random_exemplars:
            ex_set = get_dynamic_dataset(ex_data)
            len_ex = len(ex_set)
            ex_dataloader = DataLoader(ex_set,args.ex_size*args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = True)
        else:
            ex_set = get_ex_set(ex_data, args)
            len_ex = len(ex_set)
            ex_set = get_dynamic_dataset(ex_set)
            ex_dataloader = DataLoader(ex_set,args.ex_size*args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = False)
            # len_ex = len(ex_dataloader)
        ex_used = 0
        ex_dataloader = iter(ex_dataloader)

    print(f"batch_size: {args.batch_size}")
    print("starting training...")
    
    for batch in tqdm(my_dataloader, total=len(my_dataloader)):

        correctness, feats_l, feats_r = batch

        # Use packed sequences for variable-length
        # (not needed for Whisper, but needed for some others)
        feats_l = torch.nn.utils.rnn.pack_padded_sequence(
            feats_l.data, 
            (feats_l.lengths * feats_l.data.size(1)).to(torch.int64), 
            batch_first=True,
            enforce_sorted = False
        )
        feats_r = torch.nn.utils.rnn.pack_padded_sequence(
            feats_r.data, 
            (feats_r.lengths * feats_r.data.size(1)).to(torch.int64), 
            batch_first=True,
            enforce_sorted = False
        )
        feats_l = feats_l.to(args.device)
        feats_r = feats_r.to(args.device)

        # Get the exemplars for the minibatch, if using exemplar model
        if ex_data is not None:
            # print(f"ex_used: {ex_used}, len_ex: {len_ex}")
            ex_used += args.ex_size * args.num_minervas
            if ex_used > len_ex:
                if args.random_exemplars:
                    ex_dataloader = iter(DataLoader(ex_set,args.ex_size * args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = True))
                else:
                # print("Reloading exemplars...")
                    ex_set = get_ex_set(ex_data, args)
                    ex_set = get_dynamic_dataset(ex_set)
                    ex_dataloader = iter(DataLoader(ex_set,args.ex_size * args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = False))
                ex_used = args.ex_size * args.num_minervas
                
            # print(f"ex_used: {ex_used}, len_ex: {len_ex}")

            exemplars = next(ex_dataloader)
            ex_correct, ex_feats_l, ex_feats_r = exemplars
            # print(f"ex_correct:\n{ex_correct}")
            # print(f"\nexemplar correctness:\n{ex_correct}\n")
            ex_feats_l = torch.nn.utils.rnn.pack_padded_sequence(
                ex_feats_l.data, 
                (ex_feats_l.lengths * ex_feats_l.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            ex_feats_r = torch.nn.utils.rnn.pack_padded_sequence(
                ex_feats_r.data, 
                (ex_feats_r.lengths * ex_feats_r.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            ex_feats_l = ex_feats_l.to(args.device)
            ex_feats_r = ex_feats_r.to(args.device)
            ex_correct = ex_correct.data.to(args.device)

        target_scores = correctness.data.to(args.device)
        
        optimizer.zero_grad()

        if ex_data is not None:
            output_l, _ = model(X = feats_l, D = ex_feats_l, r = ex_correct, left = True, num_minervas = args.num_minervas)
            output_r, _ = model(X = feats_r, D = ex_feats_r, r = ex_correct, left = False, num_minervas = args.num_minervas)
        elif args.exemplar:
            output_l, _ = model(feats_l, left = True, num_minervas = args.num_minervas)
            output_r, _ = model(feats_r, left = False, num_minervas = args.num_minervas)
        else:
            output_l,_ = model(feats_l)
            output_r,_ = model(feats_r)
        loss_l = criterion(output_l,target_scores)
        loss_r = criterion(output_r,target_scores)

        # Sum the right and left losses - note that training loss is
        # therefore doubled compared to evaluation loss
        loss = loss_l + loss_r

        loss.backward()
        if args.grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
        optimizer.step()
        loss_list.append(loss.item())
        running_loss += loss.item()

    if args.model == "LSTM_layers" or args.model == "ExLSTM_layers":
        print(f"\nlayer weights sm:\n{model.sm(model.layer_weights)}\n")
        print(f"\nlayer weights:\n{model.layer_weights}\n")
        print(f"layer weights sum: {model.sm(model.layer_weights).sum().item()}\n")
        
    return model,optimizer,criterion,sum(loss_list)/len(loss_list)


def save_model(model,opt,epoch,args,val_loss):
    p = args.model_dir
    if not os.path.exists(p):
        os.mkdir(p)
    m_name = "%s-%s"%(args.model,args.seed)
    torch.save(model.state_dict(),"%s/%s_model.pt"%(p,m_name))
    torch.save(opt.state_dict(),"%s/%s_opt.pt"%(p,m_name))
    try:
        torch.save(model.Dl, "%s/%s_Dl.pt"%(p,m_name))
        torch.save(model.Dr, "%s/%s_Dr.pt"%(p,m_name))
        torch.save(model.r, "%s/%s_r.pt"%(p,m_name))
    except:
        pass

def save_model_old(model,opt,epoch,args,val_loss):
    p = args.model_dir
    if not os.path.exists(p):
        os.mkdir(p)
    m_name = "%s-%s"%(args.model,args.seed)
    torch.save(model.state_dict(),"%s/%s_%s_%s_model.pt"%(p,m_name,epoch,val_loss))
    torch.save(opt.state_dict(),"%s/%s_%s_%s_opt.pt"%(p,m_name,epoch,val_loss))
    try:
        torch.save(model.Dl, "%s/%s_%s_%s_Dl.pt"%(p,m_name,epoch,val_loss))
        torch.save(model.Dr, "%s/%s_%s_%s_Dr.pt"%(p,m_name,epoch,val_loss))
        torch.save(model.r, "%s/%s_%s_%s_r.pt"%(p,m_name,epoch,val_loss))
    except:
        pass


def main(args, config):
    #set up the torch objects
    print("creating model: %s"%args.feats_model)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.exemplar = False

    # Select a pretrained feature extractor model
    if args.feats_model == "XLSREncoder":
        feat_extractor = XLSREncoder_feats()
        dim_extractor = 512
        hidden_size = 512//2
        activation = nn.LeakyReLU
        att_pool_dim = 512

    elif args.feats_model == "XLSRFull":
        feat_extractor = XLSRFull_feats()
        dim_extractor = 1024
        hidden_size = 1024//2
        activation = nn.LeakyReLU
        att_pool_dim = 1024
        
    elif args.feats_model == "XLSRCombo":
        ## Tricky one! Check fat extractor
        print("XLSRCombo Not implemented.")
        quit()
        # combo = True
        # feat_extractor = XLSRCombo_feats()
        # dim_extractor = 1024
        # hidden_size = 1024//2
        # activation = nn.LeakyReLU
        
    elif args.feats_model == "HuBERTEncoder":
        feat_extractor = HuBERTEncoder_feats()
        dim_extractor = 512
        hidden_size = 512//2
        activation = nn.LeakyReLU
        att_pool_dim = 512
        print(feat_extractor)
        
    elif args.feats_model == "HuBERTFull":
        feat_extractor = HuBERTFull_feats()
        dim_extractor = 768
        hidden_size = 768//2
        activation = nn.LeakyReLU
        att_pool_dim = 768
        
    elif args.feats_model == "Spec":  
        feat_extractor = Spec_feats()
        dim_extractor = 257
        hidden_size = 257//2
        activation = nn.LeakyReLU
        att_pool_dim = 256

    elif args.feats_model == "WhisperEncoder":  
        feat_extractor = WhisperEncoder_feats(
            pretrained_model=args.pretrained_feats_model, 
            use_feat_extractor = True,
            layer = args.layer
        )
        dim_extractor = 768
        hidden_size = 768//2
        activation = nn.LeakyReLU
        att_pool_dim = 768

    elif args.feats_model == "WhisperFull":  
        feat_extractor = WhisperFull_feats(
            pretrained_model=args.pretrained_feats_model, 
            use_feat_extractor = True,
            layer = args.layer
        )
        dim_extractor = 768
        hidden_size = 768//2
        activation = nn.LeakyReLU
        att_pool_dim = 768
        
    elif args.feats_model == "WhisperBase":  
        feat_extractor = WhisperBase_feats(
            pretrained_model=args.pretrained_feats_model, 
            use_feat_extractor = True,
            layer = args.layer
        )
        dim_extractor = 512
        hidden_size = 512//2
        activation = nn.LeakyReLU
        att_pool_dim = 512
        
    else:
        print("Feats extractor not recognised")
        exit(1)


    # Make a model directory (if it doesn't exist) and write out config for future reference
    # args.model_dir = model_dir
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    with open(args.model_dir + "/config.json", 'w+') as f:
        f.write(json.dumps(dict(config)))
        
    # You can save the feats extracted to disk so it doesn't have to be done 
    # again for future test runs (takes up space though)
    pre = "" if args.pretrained_feats_model is None else "pre"
    save_feats_file = f"{args.dataroot}{args.feats_model}{pre}_N{args.N}_{'debug_' if args.debug else ''}{'' if args.layer == -1 else args.layer}"
    print(f"save_feats_file:\n{save_feats_file}")
    data = pd.read_json(args.in_json_file)
    data["subset"] = "CEC1"
    if not args.use_CPC1:
        data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
        data2["subset"] = "CEC2"
        data = pd.concat([data, data2])
    data = data.drop_duplicates(subset = ['signal'], keep = 'last')
    # print(data["correctness"])
    if args.learn_incorrect:
        data["correctness"] = 100 - data["correctness"].to_numpy()
    data["predicted"] = np.nan  

    if args.debug:
        _, data = train_test_split(data, test_size = 0.01)

    theset = "train_indep" if args.use_CPC1 else "train"
    data = get_feats(
        data = data,
        save_feats_file = save_feats_file,
        dim_extractor = dim_extractor,
        feat_extractor = feat_extractor,
        theset = theset,
        args = args
    )
    
    # Mark various disjoint validation sets in the data
    data = get_disjoint_val_set(args, data)
    dis_val_data = data[data.validation > 0].copy()
    train_data,val_data = train_test_split(data[data.validation == 0],test_size=0.1)

    if args.fix_ex:
        if args.random_exemplars:
            ex_set = get_dynamic_dataset(train_data)
            ex_dataloader = DataLoader(ex_set,args.ex_size*args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = True)
        else:
            ex_set = get_ex_set(train_data, args)
            ex_set = get_dynamic_dataset(ex_set)
            ex_dataloader = DataLoader(ex_set,args.ex_size*args.num_minervas,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = True)
            # len_ex = len(ex_dataloader)
        ex_dataloader = iter(ex_dataloader)
        ex_correct, ex_feats_l, ex_feats_r = next(ex_dataloader)
    else:
        ex_feats_l = None
        ex_feats_r = None
        ex_correct = None

    # Select classifier model (this is the one we're actually training)
    if args.model == "LSTM":
        model = MetricPredictorLSTM(dim_extractor, hidden_size, activation, att_pool_dim)
    elif args.model == "LSTM_layers":
        model = MetricPredictorLSTM_layers(dim_extractor, hidden_size, activation, att_pool_dim, num_layers = args.num_layers)
    elif args.model == "AttenPool":
        model = MetricPredictorAttenPool(att_pool_dim)
    elif args.model == "ExLSTM":
        model = ExLSTM(dim_extractor, hidden_size, att_pool_dim, p_factor = args.p_factor, minerva_dim = args.minerva_dim)
        args.exemplar = True
    elif args.model == "ExLSTM_layers":
        model = ExLSTM_layers(
            dim_extractor, 
            hidden_size, 
            att_pool_dim, 
            p_factor = args.p_factor, 
            minerva_dim = args.minerva_dim, 
            minerva_R_dim = args.minerva_r_dim,
            num_layers = args.num_layers,
            use_r_encoding = args.use_r_encoding
        )
        args.exemplar = True
    elif args.model == "ExLSTM_layers_learn":
        model = ExLSTM_layers_learn(
            dim_extractor = dim_extractor, 
            hidden_size = hidden_size, 
            att_pool_dim = att_pool_dim, 
            ex_feats_l = ex_feats_l,
            ex_feats_r = ex_feats_r,
            ex_correct = ex_correct,
            p_factor = args.p_factor, 
            minerva_dim = args.minerva_dim, 
            minerva_R_dim = args.minerva_r_dim,
            num_layers = args.num_layers,
            use_r_encoding = args.use_r_encoding
        )
        args.exemplar = True
    else:
        print("Model not recognised")
        exit(1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters]) 
    print(f"Number of parameters: {num_params}")

    # Set up logging on WandB (handy for keeping an eye on things)
    if not args.skip_wandb:
        # wandb_name = "%s_%s_%s_%s_feats_%s_%s"%(args.exp_id,args.N,args.model,ex,date,args.seed)
        run = wandb.init(
            project=args.wandb_project, 
            reinit = True, 
            name = args.model_name,
            tags = [f"N{args.N}", f"lr{args.lr}", args.feats_model, args.model, f"bs{args.batch_size}"]
        )
        if args.exemplar:
            run.tags = run.tags + (
                "exemplar", 
                f"es{args.ex_size}{args.exemplar_source}", 
                f"p{args.p_factor}",
                f"fe{int(args.fix_ex)}",
                f"tec{int(args.train_ex_class)}"
            )
        if args.restart_model is not None:
            run.tags = run.tags + ("resumed", )
        if args.layer is not None:
            run.tags = run.tags + (f"layer{args.layer}", )
        if args.minerva_dim is not None:
            run.tags = run.tags + (f"min_dim{args.minerva_dim}", )
        if args.random_exemplars:
            run.tags = run.tags + (f"random_ex", )
        else:
            run.tags = run.tags + (f"strat_ex", )
        if args.num_minervas != 1:
            run.tags = run.tags + (f"nm{args.num_minervas}", )
        if args.learn_incorrect:
            run.tags = run.tags + (f"learn_incorrect", )
        if args.minerva_r_dim is not None:
            run.tags = run.tags + (f"r_dim{args.minerva_r_dim}", )
        if args.use_r_encoding:
            run.tags = run.tags + (f"r_encoding", )
        if args.train_disjoint:
            run.tags = run.tags + (f"train_disjoint", )
    
    criterion = nn.MSELoss()

    optParams = []
    for param_name, param in model.named_parameters():
        if param_name == "layer_weights":
            optParams.append(
                {'params': param, 'weight_decay': 0}
            )
        else:
            optParams.append(
                {'params': param, 'weight_decay': args.weight_decay}
            )

    optimizer = optim.Adam(optParams,lr=args.lr)
    if not args.use_CPC1:
        if int(args.in_json_file.split("/")[-1].split(".")[-2]) != int(args.N):
            print("Warning: N does not match dataset:")
            print(args.in_json_file.split("/")[-1].split(".")[-2],args.N)
            exit()


    # If restarting a training run, load the weights
    if args.restart_model is not None:
        model.load_state_dict(torch.load(args.restart_model))


    # Reset seeds for repeatable behaviour regardless of
    # how feats / models are obtained
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Split into train and val sets
    if args.train_disjoint:
        print("Training on disjoint")
        train_data = pd.concat([train_data, dis_val_data])
    # print(dis_val_data["correctness"])
    # print(train_data["correctness"])
    # print(val_data["correctness"])
    # quit()

    print("Trainset: %s\nValset: %s\nDisValset: %s"%(train_data.shape[0],val_data.shape[0],dis_val_data.shape[0]))
    print("=====================================")

    if args.exemplar and not args.fix_ex:
        if args.exemplar_source == "CEC1":
            train_ex_data = train_data[train_data.subset == "CEC1"]
            dis_val_ex_data = train_data[train_data.subset == "CEC1"]
        elif args.exemplar_source == "CEC2":
            train_ex_data = train_data[train_data.subset == "CEC2"]
            dis_val_ex_data = train_data[train_data.subset == "CEC2"]
        elif args.exemplar_source == "matching":
            train_ex_data = train_data
            dis_val_ex_data = train_data[train_data.subset == "CEC2"]
        elif args.exemplar_source == "all":
            train_ex_data = train_data
            dis_val_ex_data = train_data
        else:
            print("Invalid exemplar exemplar source.")
            quit()
        print("Using training data for exemplars")
    else:
        train_ex_data = None
        dis_val_ex_data = None

    train_data = train_data.sample(frac=1).reset_index(drop=True)

    model = model.to(args.device)

    best_val_loss = 999

    if not args.skip_train:
        print("Starting training of model: %s\nlearning rate: %s\nseed: %s\nepochs: %s\nsave location: %s/"%(args.model,args.lr,args.seed,args.n_epochs,args.model_dir))
        print("=====================================")
        for epoch in range(args.restart_epoch, args.n_epochs + args.restart_epoch):

            model,optimizer,criterion,training_loss = train_model(model,train_data,optimizer,criterion,args,ex_data=train_ex_data)
            training_loss /= 2      # training loss is left + right so divide

            val_preds, val_loss = validate_model(model,val_data,optimizer,criterion,args,train_ex_data)
            preds, _ = validate_model(model,dis_val_data,optimizer,criterion,args,dis_val_ex_data)

            check_val_loss = criterion(torch.tensor(val_preds) / 100, torch.tensor(val_data.correctness.values) / 100).item()

            # Get losses for the various disjoint subsets held within dis_val_data
            dis_val, dis_lis_val, dis_sys_val, dis_scene_val = get_dis_val_set_losses(
                dis_val_preds = preds,
                correct = dis_val_data["correctness"].values,
                validation = dis_val_data["validation"].values,
                criterion = criterion
            )

            if not args.skip_wandb:
                log_dict = {
                    "val_rmse": val_loss**0.5,
                    "dis_val_rmse": dis_val['loss']**0.5,
                    "dis_lis_val_rmse": dis_lis_val['loss']**0.5,
                    "dis_sys_val_rmse": dis_sys_val['loss']**0.5,
                    "dis_scene_val_rmse": dis_scene_val['loss']**0.5,
                    "train_rmse": training_loss**0.5
                }
                
                wandb.log(log_dict)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model,optimizer,epoch,args,val_loss**0.5)

            torch.cuda.empty_cache()
            print("Epoch: %s"%(epoch))
            print("\tTraining Loss: %s"%(training_loss**0.5))
            print("\tValidation Loss: %s"%(val_loss**0.5))
            print("\tCheck val  Loss: %s"%(check_val_loss**0.5))
            print("\tDisjoint validation Loss: %s"%(dis_val['loss']**0.5))
            print("\tDisjoint listener validation Loss: %s"%(dis_lis_val['loss']**0.5))
            print("\tDisjoint system validation Loss: %s"%(dis_sys_val['loss']**0.5))
            print("\tDisjoint scene validation Loss: %s"%(dis_scene_val['loss']**0.5))
            print("=====================================")    

            if args.model == "LSTM_layers" or args.model == "ExLSTM_layers":
                with open(f"{args.model_dir}/layer_weights.txt", 'a') as f:
                    f.write(" ".join([str(weight) for weight in model.sm(model.layer_weights).tolist()]))
                    f.write("\n")


    if args.skip_train and args.pretrained_model_dir is not None:
        model_dir = args.pretrained_model_dir
    else:
        model_dir = args.model_dir

    print(model_dir)
    model_files = os.listdir(model_dir)

    model_files = [x for x in model_files if "model" in x]
    if len(model_files) > 1:
        model_files.sort(key=lambda x: float(x.split("_")[-2].strip(".pt")))
    
    model_file = model_files[0]
    print("Loading model:\n%s"%model_file)
    Dl_file = [x for x in model_files if "_Dl.pt" in x]
    Dr_file = [x for x in model_files if "_Dr.pt" in x]
    r_file = [x for x in model_files if "_r.pt" in x]
    model.load_state_dict(torch.load("%s/%s"%(model_dir,model_file)))
    if len(Dl_file) == 1:
        print("Loading ex features")
        Dl_file = Dl_file[0]
        Dr_file = Dr_file[0]
        model.Dl = torch.load("%s/%s"%(model_dir, Dl_file))
        model.Dr = torch.load("%s/%s"%(model_dir, Dr_file))
    if len(r_file) == 1:
        print("Loading ex classes")
        r_file = r_file[0]
        model.r = torch.load("%s/%s"%(model_dir, r_file))

    
    # get validation predictions
    val_predictions,val_loss = validate_model(model,val_data,optimizer,criterion,args,train_ex_data)
    val_error = val_loss**0.5
    val_loss = criterion(torch.tensor(val_predictions) / 100, torch.tensor(val_data.correctness.values) / 100).item()
    val_loss = val_loss**0.5
    val_stats = get_stats(val_predictions, val_data.correctness.values)
    print(f"loop val loss: {val_error}, preds val loss: {val_loss}")
    val_data["predicted"] = val_predictions
    val_data[["scene", "listener", "system", "correctness", "predicted"]].to_csv(f"{args.out_csv_file}_val_preds.csv", index=False)

    dis_val_predictions, _ = validate_model(model,dis_val_data,optimizer,criterion,args,dis_val_ex_data)
    dis_val_data["predicted"] = dis_val_predictions

    # Write a csv file of restults for each data split:
    dis_val_data[["scene", "listener", "system", "validation", "correctness", "predicted"]].to_csv(f"{args.out_csv_file}_all_dis_val_preds.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation == 7].copy()
    temp_data[["scene", "listener", "system", "correctness", "predicted"]].to_csv(f"{args.out_csv_file}_dis_val_preds.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation.isin([1, 3, 5, 7])].copy()
    temp_data[["scene", "listener", "system", "correctness", "predicted"]].to_csv(f"{args.out_csv_file}_dis_lis_val_preds.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation.isin([2, 3, 6, 7])].copy()
    temp_data[["scene", "listener", "system", "correctness", "predicted"]].to_csv(f"{args.out_csv_file}_dis_sys_val_preds.csv", index=False)
    temp_data = dis_val_data[dis_val_data.validation.isin([4, 5, 6, 7])].copy()
    temp_data[["scene", "listener", "system", "correctness", "predicted"]].to_csv(f"{args.out_csv_file}_dis_scene_val_preds.csv", index=False)

    dis_val, dis_lis_val, dis_sys_val, dis_scene_val = get_dis_val_set_losses(
        dis_val_preds = dis_val_predictions,
        correct = dis_val_data["correctness"].values,
        validation = dis_val_data["validation"].values,
        criterion = criterion,
        include_stats = True
    )
    
    #normalise the predictions
    val_gt = val_data["correctness"].to_numpy()/100
    val_predictions = np.asarray(val_predictions)/100

    def logit_func(x,a,b):
     return 1/(1+np.exp(a*x+b))

    # logistic mapping curve fit to get the a and b parameters
    popt,_ = curve_fit(logit_func, val_predictions, val_gt)
    a,b = popt
    print("a: %s b: %s"%(a,b))
    print("=====================================")
    # Test the model
    if not args.skip_test:
        print("Testing model on test set")
        test_data = pd.read_json(args.test_json_file)
        # test_data["correctness"] = np.nan
        test_data["subset"] = "CEC2"
        save_feats_file = f"{args.dataroot}{args.feats_model}{pre}_N{args.N}_{'debug_' if args.debug else ''}{'' if args.layer == -1 else args.layer}_test"

        test_data = get_feats(
            data = test_data,
            save_feats_file = save_feats_file,
            dim_extractor = dim_extractor,
            feat_extractor = feat_extractor,
            theset = "test",
            args = args
        )

        eval_predictions, evalLoss = validate_model(model, test_data, optimizer, criterion, args, dis_val_ex_data)
        eval_stats = get_stats(eval_predictions, test_data.correctness.values)
        # predictions_fitted = np.asarray(predictions)/100
        #apply the logistic mapping
        # predictions_fitted = logit_func(eval_predictions,a,b)
        test_data["predicted"] = eval_predictions
        # test_data["predicted_fitted"] = predictions_fitted * 100
        test_data[["signal", "scene", "listener", "system", "correctness", "predicted"]].to_csv(f"{args.out_csv_file}_test_preds.csv", index=False)
        print(f"Evaluation loss: {evalLoss**0.5}")


    # Write the final stastistics to the summary file
    with open (args.summ_file, "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            args.out_csv_file.split("/")[-1].strip(".csv"), 
            val_error, val_stats["p_corr"], val_stats["s_corr"], val_stats["std"], 
            dis_val["loss"]**0.5, dis_val["p_corr"], dis_val["s_corr"], dis_val["std"], 
            dis_lis_val["loss"]**0.5, dis_lis_val["p_corr"], dis_lis_val["s_corr"], dis_lis_val["std"], 
            dis_sys_val["loss"]**0.5, dis_sys_val["p_corr"], dis_sys_val["s_corr"], dis_sys_val["std"], 
            dis_scene_val["loss"]**0.5, dis_scene_val["p_corr"], dis_scene_val["s_corr"], dis_scene_val["std"],
            evalLoss**0.5, eval_stats["p_corr"], eval_stats["s_corr"], eval_stats["std"]
        ])
    
    print("=====================================")

    if not args.skip_wandb:
        run.finish()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", help="location of configuation json file", default = None #"configs/default_config.json"
    )
    parser.add_argument(
        "--exp_id", help="id for individual experiment", default = None
    )
    parser.add_argument(
        "--summ_file", help="path to write summary results to" , default="save/CSL_CPC2.csv"
    )
    parser.add_argument(
        "--out_csv_file", help="path to write the preditions to" , default=None
    )
    parser.add_argument(
        "--wandb_project", help="W and B project name" , default="CSL_CPC2"
    )
    parser.add_argument(
        "--skip_wandb", help="skip logging via WandB", default=False, action='store_true'
    )
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
        "--skip_test", help="do predictions on test data", default=False, action='store_true'
    )


    # Training data
    parser.add_argument(
        "--skip_train", help="do training", default=False, action='store_true'
    )
    parser.add_argument(
        "--use_CPC1", help="train and evaluate on CPC1 data" , default=False, action='store_true'
    )
    parser.add_argument(
        "--N", help="train split" , default=1, type=int
    )
    parser.add_argument(
        "--learn_incorrect", help="predict incorrectness (100 - correctness) when training", default=False, action='store_true'
    )
    parser.add_argument(
        "--train_disjoint", help="include the disjoin val data in training data (for final model only)", default=False, action='store_true'
    )

    # Feats details
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
        "--fix_ex", help="use fixed exemplars", default=False, action='store_true'
    )
    parser.add_argument(
        "--train_ex_class", help="train the exemplar classes", default=False, action='store_true'
    )
    parser.add_argument(
        "--exemplar_source", help="one of CEC1, CEC2, all or matched", default="CEC2"
    )
    parser.add_argument(
        "--use_r_encoding", help="use positional encoding for exemplar correctness", default=False, action='store_true'
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
    config["skip_train"] = args.skip_train
    config["use_CPC1"] = args.use_CPC1
    config["N"] = args.N
    config["skip_test"] = args.skip_test

    if args.feats_model is None:
        args.feats_model = config["feats_model"] if "feats_model" in config else "WhisperFull"
    config["feats_model"] = args.feats_model

    if args.pretrained_feats_model is None:
        args.pretrained_feats_model = config["pretrained_feats_model"] if "pretrained_feats_model" in config else None
    config["pretrained_feats_model"] = args.pretrained_feats_model

    args.pretrained_model_dir = f"save/{args.pretrained_model_dir}"
    config["pretrained_model_dir"] = args.pretrained_model_dir

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
        args.batch_size = config["batch_size"] if "batch_size" in config else 8
    config["batch_size"] = args.batch_size

    if args.lr is None:
        args.lr = config["lr"] if "lr" in config else 0.001
    config["lr"] = args.lr

    if args.weight_decay is None:
        args.weight_decay = config["weight_decay"] if "weight_decay" in config else 0.001
    config["weight_decay"] = args.weight_decay

    if args.n_epochs is None:
        args.n_epochs = config["n_epochs"] if "n_epochs" in config else 25
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

    if args.minerva_r_dim is None:
        args.minerva_r_dim = config["minerva_r_dim"] if "minerva_r_dim" in config else None
    config["minerva_r_dim"] = args.minerva_r_dim
    
    if args.num_minervas is None:
        args.num_minervas = config["num_minervas"] if "num_minervas" in config else 1
    config["num_minervas"] = args.num_minervas
    
    config["random_exemplars"] = args.random_exemplars
    config["exemplar_source"] = args.exemplar_source
    config["learn_incorrect"] = args.learn_incorrect
    config["use_r_encoding"] = args.use_r_encoding
    config["train_disjoint"] = args.train_disjoint

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
        args.wandb_project = "CPC_paper" if args.wandb_project is None else args.wandb_project
        config["wandb_project"] = args.wandb_project
        args.in_json_file = f"{DATAROOT}/metadata/CEC1.train.{args.N}.json"
        config["in_json_file"] = args.in_json_file
        if args.skip_test:
            args.test_json_file = None
            config["test_json_file"] = None
        else:
            args.test_json_file = f"{DATAROOT}/metadata/CEC2.test.{args.N}.json"
            config["test_json_file"] = args.test_json_file
        
    if args.summ_file is None:
        if args.use_CPC1:
            args.summ_file = "save/CPC1_metrics.csv"
        else:
            args.summ_file = "save/CPC_metrics.csv"
    config["summ_file"] = args.summ_file

    today = datetime.datetime.today()
    date = today.strftime("%H-%M-%d-%b-%Y")
    nm = "" if args.num_minervas == 1 else f"nm{args.num_minervas}"
    args.model_name = "%s_%s_%s_%s_%s_%s_%s"%(args.exp_id,args.N,args.feats_model,args.model,nm,date,args.seed)
    args.model_dir = "save/%s"%(args.model_name)
    config["model_name"] = args.model_name

    if args.out_csv_file is None:
        nm = "" if args.num_minervas == 1 else f"_nm{args.num_minervas}_{args.seed}"
        args.out_csv_file = f"{args.model_dir}/{args.exp_id}_N{args.N}_{args.feats_model}_{args.model}{nm}"
    config["out_csv_file"] = args.out_csv_file

    main(args, config)
