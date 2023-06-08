
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

# from models.huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
# from models.wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
# from models.llama_wrapper import LlamaWrapper
# from models.ni_predictors import MetricPredictor, MetricPredictorCombo
from models.ni_predictor_models import MetricPredictor, MetricPredictorCombo
from models.ni_feat_extractors import Spec_feats, XLSREncoder_feats, XLSRFull_feats, XLSRCombo_feats, HuBERTEncoder_feats, HuBERTFull_feats

DATAROOT = "/store/store1/data/clarity_CPC2_data/" 
# DATAROOT = "~/exp/data/clarity_CPC2_data/" 
df_listener = pd.read_json(DATAROOT + "clarity_data/metadata/listeners.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

resampler = T.Resample(32000, 16000).to(device)

def compute_feats(wavs,fs):
        """Feature computation pipeline"""
        #sum stero channels 
        #wavs = wavs[:,:,0] + wavs[:,:,1]
        #resample to 16000 for input to model
        #print(wavs.shape)
        wavs_l = wavs[:,:,0]
        wavs_r = wavs[:,:,1]
        wavs_l = resampler(wavs_l)
        wavs_r = resampler(wavs_r)
        #print(wavs_l.shape,wavs_r.shape)
        return wavs_l,wavs_r

def audio_pipeline(path,fs=32000):
    wavs = sb.dataio.dataio.read_audio_multichannel(path)    
    return wavs


def format_correctness(y):
    #convert correctness percentage to tensor
    y = torch.tensor([y])
    # normalize
    y = y/100
    return y

def format_feats(y):

    y = torch.from_numpy(y).to(torch.float)
    return y[0]



def get_mean(scores):
    out_list = []
    for el in scores:
        el = el.strip("[").strip("]").split(",")
        el = [float(a) for a in el]
        #print(el,type(el))
        out_list.append(el)
    return torch.Tensor(out_list).unsqueeze(0)

def validate_model(model,test_data,optimizer,criterion,N,combo):
    out_list = []
    model.eval()
    name_list = test_data["signal"]
    correctness_list = test_data["correctness"]
    listener_list = test_data["listener"]
    scene_list = test_data["scene"]
    subset_list = test_data["subset"]
    if combo:
        feats_full_l_list = test_data['feats_full_l']
        feats_full_r_list = test_data['feats_full_r']
        feats_extract_l_list = test_data['feats_extract_l']
        feats_extract_r_list = test_data['feats_extract_r']
    else:
        feats_l_list = test_data["feats_l"]
        feats_r_list = test_data["feats_r"]

    #haspi_list = test_data["haspi"]
    #print(name_list)
    running_loss = 0.0
    loss_list = []
    test_dict = {}
    
    train_dict = {}
    if combo:
        for sub,name,corr,scene,lis, f_full_l, f_full_r, f_extract_l, f_extract_r in  zip(subset_list,name_list,correctness_list,scene_list,listener_list, feats_full_l_list, feats_full_r_list, feats_extract_l_list, feats_extract_r_list):
            
            if name in test_dict:
                test_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_full_l":f_full_l, "feats_full_r": f_full_r, "feats_extract_l": f_extract_l, "feats_extract_r": f_extract_r}
            else:
                test_dict[name + "_" + sub] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_full_l":f_full_l, "feats_full_r": f_full_r, "feats_extract_l": f_extract_l, "feats_extract_r": f_extract_r}

        dynamic_items = [
            {"func": lambda l: format_correctness(l),
            "takes": "correctness",
            "provides": "formatted_correctness"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_full_l",
            "provides": "formatted_feats_full_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_full_r",
            "provides": "formatted_feats_full_r"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_extract_l",
            "provides": "formatted_feats_extract_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_extract_r",
            "provides": "formatted_feats_extract_r"},
            # {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
            # "takes": ["signal","subset"],
            # "provides": "wav"},
        ]
        
        test_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
        #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
        test_set.set_output_keys(["formatted_correctness", "formatted_feats_full_l", "formatted_feats_full_r", "formatted_feats_extract_l", "formatted_feats_extract_r"])
    else:
        for sub,name,corr,scene,lis, f_l, f_r in  zip(subset_list, name_list,correctness_list,scene_list,listener_list, feats_l_list, feats_r_list):
            # Bodge job correction - one name is in both CEC1 and CEC2 subsets, leading to a mismatch in the length
            # of dataset and predictions
            if name in test_dict:
                test_dict[name + "_" + sub] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_l":f_l, "feats_r": f_r}
            else:
                test_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_l":f_l, "feats_r": f_r}
        
        dynamic_items = [
            {"func": lambda l: format_correctness(l),
            "takes": "correctness",
            "provides": "formatted_correctness"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_l",
            "provides": "formatted_feats_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_r",
            "provides": "formatted_feats_r"},
            # {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
            # "takes": ["signal","subset"],
            # "provides": "wav"},
        ]
        test_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
        test_set.set_output_keys(["formatted_correctness", "formatted_feats_l", "formatted_feats_r"])

    my_dataloader = DataLoader(test_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting validation...")
    for batch in tqdm(my_dataloader):
       
        batch = batch.to(device)
        if combo:
            correctness, feats_full_l, feats_full_r, feats_extract_l, feats_extract_r = batch
            feats_full_l = feats_full_l.data
            feats_full_r = feats_full_r.data
            feats_extract_l = feats_extract_l.data
            feats_extract_r = feats_extract_r.data
        else:
            correctness, feats_l, feats_r = batch
            feats_l = feats_l.data
            feats_r = feats_r.data
            
        correctness = correctness.data
        target_scores = correctness
        
        if combo:
            output_l,_ = model(feats_full_l.float(), feats_extract_l.float())
            output_r,_ = model(feats_full_r.float(), feats_extract_r.float())
        else:
            output_l,_ = model(feats_l.float())
            output_r,_ = model(feats_r.float())
            
        output = max(output_l,output_r)
        loss = criterion(output,target_scores)

        out_list.append(output.detach().cpu().numpy()[0][0]*100)

        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item()
    return out_list,sum(loss_list)/len(loss_list)

def test_model(model,test_data,optimizer,criterion,N,combo):
    out_list = []
    model.eval()
    name_list = test_data["signal"]
    correctness_list = test_data["correctness"]
    listener_list = test_data["listener"]
    scene_list = test_data["scene"]
    if combo:
        feats_full_l_list = test_data['feats_full_l']
        feats_full_r_list = test_data['feats_full_r']
        feats_extract_l_list = test_data['feats_extract_l']
        feats_extract_r_list = test_data['feats_extract_r']
    else:
        feats_l_list = test_data["feats_l"]
        feats_r_list = test_data["feats_r"]
    #haspi_list = test_data["haspi"]
    #print(name_list)
    running_loss = 0.0
    loss_list = []
    test_dict = {}
    if combo:
        for name,corr,scene,lis, f_full_l, f_full_r, f_extract_l, f_extract_r in  zip(name_list,correctness_list,scene_list,listener_list, feats_full_l_list, feats_full_r_list, feats_extract_l_list, feats_extract_r_list):
            test_dict[name] = {"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_full_l":f_full_l, "feats_full_r": f_full_r, "feats_extract_l": feats_extract_l_list, "feats_extract_r": feats_extract_r_list}

        dynamic_items = [
            {"func": lambda l: format_correctness(l),
            "takes": "correctness",
            "provides": "formatted_correctness"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_full_l",
            "provides": "formatted_feats_full_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_full_r",
            "provides": "formatted_feats_full_r"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_extract_l",
            "provides": "formatted_feats_extract_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_extract_r",
            "provides": "formatted_feats_extract_r"},
            # {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
            # "takes": ["signal","subset"],
            # "provides": "wav"},
        ]
        
        train_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
        #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
        train_set.set_output_keys(["formatted_correctness", "formatted_feats_full_l", "formatted_feats_full_r", "formatted_feats_extract_l", "formatted_feats_extract_r"])
    else:
        for name,corr,scene,lis, f_l, f_r in  zip(name_list,correctness_list,scene_list,listener_list, feats_l_list, feats_r_list):
            test_dict[name] = {"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_l":f_l, "feats_r": f_r}

        dynamic_items = [
            {"func": lambda l: format_correctness(l),
            "takes": "correctness",
            "provides": "formatted_correctness"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_l",
            "provides": "formatted_feats_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_r",
            "provides": "formatted_feats_r"},
            # {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
            # "takes": ["signal","subset"],
            # "provides": "wav"},
        ]
        test_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
        #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
        test_set.set_output_keys(["formatted_correctness", "formatted_feats_l", "formatted_feats_r"])

    my_dataloader = DataLoader(test_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting testing...")
    for batch in tqdm(my_dataloader):

        batch = batch.to(device)
        if combo:
            correctness, feats_full_l, feats_full_r, feats_extract_l, feats_extract_r = batch
            feats_full_l = feats_full_l.data
            feats_full_r = feats_full_r.data
            feats_extract_l = feats_extract_l.data
            feats_extract_r = feats_extract_r.data
        else:
            correctness, feats_l, feats_r = batch
            feats_l = feats_l.data
            feats_r = feats_r.data

        correctness =correctness.data
        target_scores = correctness
        
        if combo:
            output_l,_ = model(feats_full_l.float(), feats_extract_l.float())
            output_r,_ = model(feats_full_r.float(), feats_extract_r.float())
        else:
            output_l,_ = model(feats_l.float())
            output_r,_ = model(feats_r.float())

        output = max(output_l,output_r)
        loss = criterion(output,target_scores)

        out_list.append(output.detach().cpu().numpy()[0][0]*100)

        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item()

    return out_list,sum(loss_list)/len(loss_list)

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
    



def train_model(model,train_data,optimizer,criterion,N,combo=False):
    model.train()
    name_list = train_data["signal"]
    correctness_list = train_data["correctness"]
    #haspi_list = train_data["HASPI"]
    scene_list = train_data["scene"]
    listener_list = train_data["listener"]
    subset_list = train_data["subset"]
    if combo:
        feats_full_l_list = train_data['feats_full_l']
        feats_full_r_list = train_data['feats_full_r']
        feats_extract_l_list = train_data['feats_extract_l']
        feats_extract_r_list = train_data['feats_extract_r']
    else:
        feats_l_list = train_data["feats_l"]
        feats_r_list = train_data["feats_r"]
        
    running_loss = 0.0
    loss_list = []
   
    train_dict = {}
    if combo:
        for sub,name,corr,scene,lis, f_full_l, f_full_r, f_extract_l, f_extract_r in  zip(subset_list,name_list,correctness_list,scene_list,listener_list, feats_full_l_list, feats_full_r_list, feats_extract_l_list, feats_extract_r_list):
            train_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_full_l":f_full_l, "feats_full_r": f_full_r, "feats_extract_l": f_extract_l, "feats_extract_r": f_extract_r}

        dynamic_items = [
            {"func": lambda l: format_correctness(l),
            "takes": "correctness",
            "provides": "formatted_correctness"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_full_l",
            "provides": "formatted_feats_full_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_full_r",
            "provides": "formatted_feats_full_r"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_extract_l",
            "provides": "formatted_feats_extract_l"},
            {"func": lambda l: format_feats(l),
            "takes": "feats_extract_r",
            "provides": "formatted_feats_extract_r"},
            # {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
            # "takes": ["signal","subset"],
            # "provides": "wav"},
        ]
        
        train_set = sb.dataio.dataset.DynamicItemDataset(train_dict,dynamic_items)
        #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
        train_set.set_output_keys(["formatted_correctness", "formatted_feats_full_l", "formatted_feats_full_r", "formatted_feats_extract_l", "formatted_feats_extract_r"])
    else:
        for sub,name,corr,scene,lis, f_l, f_r in  zip(subset_list,name_list,correctness_list,scene_list,listener_list, feats_l_list, feats_r_list):
            train_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_l":f_l, "feats_r": f_r}

            dynamic_items = [
                {"func": lambda l: format_correctness(l),
                "takes": "correctness",
                "provides": "formatted_correctness"},
                {"func": lambda l: format_feats(l),
                "takes": "feats_l",
                "provides": "formatted_feats_l"},
                {"func": lambda l: format_feats(l),
                "takes": "feats_r",
                "provides": "formatted_feats_r"},
                # {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
                # "takes": ["signal","subset"],
                # "provides": "wav"},
            ]

        train_set = sb.dataio.dataset.DynamicItemDataset(train_dict,dynamic_items)
        #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
        train_set.set_output_keys(["formatted_correctness", "formatted_feats_l", "formatted_feats_r"])

    my_dataloader = DataLoader(train_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting training...")
    
    for batch in tqdm(my_dataloader, total=len(my_dataloader)):
        batch = batch.to(device)
        if combo:
            correctness, feats_full_l, feats_full_r, feats_extract_l, feats_extract_r = batch
            feats_full_l = feats_full_l.data
            feats_full_r = feats_full_r.data
            feats_extract_l = feats_extract_l.data
            feats_extract_r = feats_extract_r.data
        else:
            correctness, feats_l, feats_r = batch
            feats_l = feats_l.data
            feats_r = feats_r.data
        #print("wavs:%s\n correctness:%s\n"%(wavs.data.shape,correctness))
        correctness =correctness.data
        target_scores = correctness
        
        # #print(wavs_data.shape)
        # feats_l,feats_r = compute_feats(wavs_data,44100) 
    
        optimizer.zero_grad()
        if combo:
            output_l,_ = model(feats_full_l.float(), feats_extract_l.float())
            output_r,_ = model(feats_full_r.float(), feats_extract_r.float())
        else:
            output_l,_ = model(feats_l.float())
            output_r,_ = model(feats_r.float())
        loss_l = criterion(output_l,target_scores)
        loss_r = criterion(output_r,target_scores)
        loss = loss_l + loss_r

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        running_loss += loss.item()
        
        
    #print("Average Training loss: %s"%(sum(loss_list)/len(loss_list)))
    
    return model,optimizer,criterion,sum(loss_list)/len(loss_list)


# def convert_audiogram(listener):
#     audiogram_l =  format_audiogram(np.array(df_listener[listener]["audiogram_levels_l"]))
#     audiogram_r =  format_audiogram(np.array(df_listener[listener]["audiogram_levels_r"]))
#     audiogram = [audiogram_l,audiogram_r]
#     return audiogram



def save_model(model,opt,epoch,args,val_loss):
    p = args.model_dir
    if not os.path.exists(p):
        os.mkdir(p)
    m_name = "%s-%s"%(args.model,args.seed)
    torch.save(model.state_dict(),"%s/%s_%s_%s_model.pt"%(p,m_name,epoch,val_loss))
    torch.save(opt.state_dict(),"%s/%s_%s_%s_opt.pt"%(p,m_name,epoch,val_loss))


def extract_feats(feat_extractor, data, N, combo = False):
    feat_extractor.eval()
    name_list = data["signal"]
    correctness_list = data["correctness"]
    listener_list = data["listener"]
    scene_list = data["scene"]
    subset_list = data["subset"]

    test_dict = {}
    
    for sub,name,corr,scene,lis in zip(subset_list,name_list,correctness_list,scene_list,listener_list):
        # Bodge job correction - one name is in both CEC1 and CEC2 subsets, leading to a mismatch in the length
        # of dataset and predictions
        if name in test_dict:
            test_dict[name + "_" + sub] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis}
        else:
            test_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis}
    
    dynamic_items = [
        {"func": lambda l: format_correctness(l),
        "takes": "correctness",
        "provides": "formatted_correctness"},
        {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
        "takes": ["signal","subset"],
        "provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/train/%s_HL-output.wav"%(DATAROOT,l),44100),
        #"takes": "signal",
        #"provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/scenes//%s_target_anechoic.wav"%(DATAROOT,l),44100),
        #"takes": "scene",
        #"provides": "clean_wav"},
    ]
    data_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
    #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
    data_set.set_output_keys(["wav", "formatted_correctness"])

    if combo:
        feats_list_full_l = []
        feats_list_extract_l = []
        feats_list_full_r = []
        feats_list_extract_r = []
    else:
        feats_list_l = []
        feats_list_r = []
        correct_list = []

    my_dataloader = DataLoader(data_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("Extracting features...")
    for batch in tqdm(my_dataloader):
        batch = batch.to(device)
        wavs,correctness = batch
        correctness =correctness.data
        wavs_data = wavs.data
        #print("wavs:%s\n correctness:%s\naudiogram:%s"%(wavs.data.shape,correctness,audiogram))
        # target_scores = correctness
        #feats = wavs_data
        feats_l,feats_r = compute_feats(wavs_data,44100) 
        #print(feats.shape)
       
        if combo:
            extracted_feats_full_l, extracted_feats_extract_l = feat_extractor(feats_l.float())
            extracted_feats_full_r, extracted_feats_extract_r = feat_extractor(feats_r.float())

            feats_list_full_l.append(extracted_feats_full_l.detach().cpu().numpy())
            feats_list_full_r.append(extracted_feats_full_r.detach().cpu().numpy())
            feats_list_extract_l.append(extracted_feats_extract_l.detach().cpu().numpy())
            feats_list_extract_r.append(extracted_feats_extract_r.detach().cpu().numpy())

        else:

            extracted_feats_l = feat_extractor(feats_l.float())
            extracted_feats_r = feat_extractor(feats_r.float())
            # print(f"extracted_feats_l.size()\n{extracted_feats_l.size()}")
            # print(f"extracted_feats_l.size()\n{extracted_feats_l.size()}")

            feats_list_l.append(extracted_feats_l.detach().cpu().numpy())
            feats_list_r.append(extracted_feats_r.detach().cpu().numpy())
            correct_list.append(correctness)

    if combo:
        data['feats_full_l'] = feats_list_full_l
        data['feats_full_r'] = feats_list_full_r
        data['feats_extract_l'] = feats_list_extract_l
        data['feats_extract_r'] = feats_list_extract_r
    else:
        data['feats_l'] = feats_list_l
        data['feats_r'] = feats_list_r


    return data



def main(args):
    #set up the torch objects
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("creating model: %s"%args.model)
    torch.manual_seed(args.seed)
    N = args.N
    combo = False
    if args.model == "XLSREncoder":
        feat_extractor = XLSREncoder_feats()
        model = MetricPredictor(dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim = 512)
        # from models.ni_predictors import XLSRMetricPredictorEncoder
        # model = XLSRMetricPredictorEncoder().to(args.device)
    elif args.model == "XLSRFull":
        feat_extractor = XLSRFull_feats()
        model = MetricPredictor(dim_extractor=1024, hidden_size=1024//2, activation=nn.LeakyReLU, att_pool_dim=1024)
        # from models.ni_predictors import XLSRMetricPredictorFull
        # model = XLSRMetricPredictorFull().to(args.device)
    elif args.model == "XLSRCombo":
        ## Tricky one! Check fat extractor
        combo = True
        feat_extractor = XLSRCombo_feats()
        model = MetricPredictorCombo(dim_extractor=1024, hidden_size=1024//2, activation=nn.LeakyReLU)
        # from models.ni_predictors import XLSRMetricPredictorCombo
        # model = XLSRMetricPredictorCombo().to(args.device)
    elif args.model == "HuBERTEncoder":
        feat_extractor = HuBERTEncoder_feats()
        model = MetricPredictor(dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim=512)
        # from models.ni_predictors import HuBERTMetricPredictorEncoder
        # model = HuBERTMetricPredictorEncoder().to(args.device)
    elif args.model == "HuBERTFull":
        feat_extractor = HuBERTFull_feats()
        model = MetricPredictor(dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU, att_pool_dim=768)
        # from models.ni_predictors import HuBERTMetricPredictorFull
        # model = HuBERTMetricPredictorFull().to(args.device)
    elif args.model == "Spec":  
        feat_extractor = Spec_feats()
        model = MetricPredictor(dim_extractor=257, hidden_size=257//2, activation=nn.LeakyReLU, att_pool_dim = 256)
        # from models.ni_predictors import SpecMetricPredictor
        # model = SpecMetricPredictor().to(args.device)
    else:
        print("Model not recognised")
        exit(1)

    feat_extractor.eval()
    feat_extractor.requires_grad_(False)
    model = model.to(args.device)
    #torchinfo.summary(model,(1,16000*8))
    #set up the model directory
    today = datetime.datetime.today()
    date = today.strftime("%H-%M-%d-%b-%Y")

    if "indep" in args.in_json_file:
        model_dir = "save/CPC2_train%s_%s_feats_%s_%s_indep"%(args.N,args.model,date,args.seed)
        if not args.skip_wandb:
            wandb_name = "%s_%s_feats_%s_%s_indep"%(args.N,args.model,date,args.seed)
            run = wandb.init(project=args.wandb_project, reinit = True, name = wandb_name)
    else:
        model_dir = "save/CPC2_train%s_%s_feats_%s_%s"%(args.N,args.model,date,args.seed)
        if not args.skip_wandb:
            wandb_name = "%s_%s_feats_%s_%s"%(args.N,args.model,date,args.seed)
            run = wandb.init(project=args.wandb_project, reinit = True, name = wandb_name)
    args.model_dir = model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    if int(args.in_json_file.split("/")[-1].split(".")[-2]) != args.N:
        print("Warning: N does not match dataset:")
        print(args.in_json_file.split("/")[-1].split(".")[-2],args.N)
        exit()

    # Load the intelligibility data
    data = pd.read_json(args.in_json_file)
    data["subset"] = "CEC1"
    data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
    data2["subset"] = "CEC2"
    data = pd.concat([data,data2])
    data["predicted"] = np.nan  # Add column to store intel predictions

    feat_extractor.to(args.device)
    data = extract_feats(feat_extractor, data, N, combo)
    feat_extractor.to('cpu')

    print(data[:50])
    

    # Split into train and val sets
    train_data,val_data = train_test_split(data,test_size=0.1)
    #TODO set up validation to partition on scene
    if args.test_json_file is not None:
        test_data = pd.read_json(args.test_json_file)
        test_data["predicted"] = np.nan
    else:
        test_data = val_data
    print("Trainset: %s\nValset: %s\nTestset: %s "%(train_data.shape[0],val_data.shape[0],test_data.shape[0]))
    print("=====================================")

    # shuffle the training data
    train_data = train_data.sample(frac=1).reset_index(drop=True)

      
    # use this line for testing :) 
    #train_data = train_data[:50]
    #test_data = test_data[:50]
    #val_data = val_data[:50]
    if args.do_train:
        print("Starting training of model: %s\nlearning rate: %s\nseed: %s\nepochs: %s\nsave location: %s/"%(args.model,args.lr,args.seed,args.n_epochs,args.model_dir))
        print("=====================================")
        for epoch in range(args.n_epochs):


            model,optimizer,criterion,training_loss = train_model(model,train_data,optimizer,criterion,args.N,combo)

            _,val_loss = validate_model(model,val_data,optimizer,criterion,args.N,combo)

            if not args.skip_wandb:
                
                wandb.log({
                    "dev_loss": val_loss,
                    "train_loss": training_loss
                })
            
            save_model(model,optimizer,epoch,args,val_loss)
            torch.cuda.empty_cache()
            print("Epoch: %s | Training Loss: %s | Validation Loss: %s"%(epoch,training_loss,val_loss))
            print("=====================================")    
    print(model_dir)
    model_files = os.listdir(model_dir)

    model_files = [x for x in model_files if "model" in x]
    #print(model_files)
    model_files.sort(key=lambda x: float(x.split("_")[-2].strip(".pt")))
    model_file = model_files[0]
    print("Loading model:\n%s"%model_file)
    model.load_state_dict(torch.load("%s/%s"%(model_dir,model_file)))
    
    # get validation predictions
    val_predictions,_ = validate_model(model,val_data,optimizer,criterion,args.N,combo)
    
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
    if args.test_json_file is not None:
        print("Testing model on test set")
        predictions,test_loss = test_model(model,test_data,optimizer,criterion,args.N)
        predictions_fitted = np.asarray(predictions)/100
        #apply the logistic mapping
        predictions_fitted = logit_func(predictions_fitted,a,b)
        test_data["predicted"] = predictions
        test_data["predicted_fitted"] = predictions_fitted*100

        test_data[["scene", "listener", "system", "predicted","predicted_fitted"]].to_csv(args.out_csv_file, index=False)


    else:
        print("Testing model on train+val set")
        predictions,test_loss = validate_model(model,data,optimizer,criterion,args.N,combo)
    
        data["predicted"] = predictions
        data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file, index=False)
    print("Test Loss: %s"%test_loss)
    print("=====================================")

    
    if not args.skip_wandb:
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_json_file", help="JSON file containing the CPC2 training metadata"
    )
    parser.add_argument(
        "out_csv_file", help="output csv file containing the intelligibility predictions"
    )
    parser.add_argument(
        "N", help="partion on which to train/test", type=int
    )
    parser.add_argument(
        "--test_json_file", help="JSON file containing the CPC2 test metadata",
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=50, type=int
    )
    parser.add_argument(
        "--lr", help="learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--model", help="model type" , default="XLSREncoder",
    )
    parser.add_argument(
        "--seed", help="torch seed" , default=1234,
    )
    parser.add_argument(
        "--do_train", help="do training" , default=True,type=bool
    )
    parser.add_argument(
        "--skip_wandb", help="skip logging via WandB" , default=False, type=bool
    )
    parser.add_argument(
        "--wandb_project", help="WandB project name", default = "CPC2"
    )
    args = parser.parse_args()
    main(args)
