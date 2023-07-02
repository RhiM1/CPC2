
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
from disjoint_val import get_disjoint_val_set

# from models.huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
# from models.wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
# from models.llama_wrapper import LlamaWrapper
# from models.ni_predictors import MetricPredictor, MetricPredictorCombo
from models.ni_predictor_models import MetricPredictorLSTM, MetricPredictorLSTMCombo, MetricPredictorAttenPool
from models.ni_feat_extractors import Spec_feats, XLSREncoder_feats, XLSRFull_feats, \
    XLSRCombo_feats, HuBERTEncoder_feats, HuBERTFull_feats, WhisperEncoder_feats
from models.ni_predictor_exemplar_models import ExemplarMetricPredictor, ExemplarMetricPredictorCombo

DATAROOT = "/store/store1/data/clarity_CPC2_data/" 
DATAROOT_CPC1 = "/store/store1/data/clarity_CPC1_data/" 
# DATAROOT = "~/exp/data/clarity_CPC2_data/" 
# df_listener = pd.read_json(DATAROOT + "clarity_data/metadata/listeners.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

resampler = T.Resample(32000, 16000).to(device)

def rmse_score(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def std_err(x, y):
    return np.std(x - y) / np.sqrt(len(x))

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


def extract_feats(feat_extractor, data, args, theset, combo = False):
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
        
    ]
    if args.use_CPC1:
        dynamic_items.append(
            {"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/%s/%s.wav"%(args.dataroot,theset,l),32000),
            "takes": ["signal"],
            "provides": "wav"}
        )
    else:
        dynamic_items.append(
            {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/%s.%s/%s/%s.wav"%(args.dataroot,theset,args.N,y,l),32000),
            "takes": ["signal","subset"],
            "provides": "wav"}
        )

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
        feats_l,feats_r = compute_feats(wavs_data,44100)
       
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

def make_disjoint_train_set(
    full_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Make a disjoint train set for given test samples."""
    # make sure that the train and test sets are disjoint
    # i.e. no signals, systems or listeners are shared
    train_df = full_df[~full_df.signal.isin(test_df.signal)]
    train_df = train_df[~train_df.system.isin(test_df.system)]
    train_df = train_df[~train_df.listener.isin(test_df.listener)]
    assert not set(train_df.signal).intersection(set(test_df.signal))
    return train_df



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
        if name in data_dict:
            data_dict[name + "_" + sub] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_l":f_l, "feats_r": f_r}
        else:
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


def get_dynamic_dataset_combo(data):

    name_list = data["signal"]
    correctness_list = data["correctness"]
    #haspi_list = train_data["HASPI"]
    scene_list = data["scene"]
    listener_list = data["listener"]
    subset_list = data["subset"]
    feats_full_l_list = data['feats_full_l']
    feats_full_r_list = data['feats_full_r']
    feats_extract_l_list = data['feats_extract_l']
    feats_extract_r_list = data['feats_extract_r']
   
    data_dict = {}
    for sub,name,corr,scene,lis, f_full_l, f_full_r, f_extract_l, f_extract_r in  zip(subset_list,name_list,correctness_list,scene_list,listener_list, feats_full_l_list, feats_full_r_list, feats_extract_l_list, feats_extract_r_list):
        if name in data_dict:
            data_dict[name + "_" + sub] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_full_l":f_full_l, "feats_full_r": f_full_r, "feats_extract_l": f_extract_l, "feats_extract_r": f_extract_r}
        else:
            data_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis, "feats_full_l":f_full_l, "feats_full_r": f_full_r, "feats_extract_l": f_extract_l, "feats_extract_r": f_extract_r}
    
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
    ]
    
    ddata = sb.dataio.dataset.DynamicItemDataset(data_dict,dynamic_items)
    #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
    ddata.set_output_keys(["formatted_correctness", "formatted_feats_full_l", "formatted_feats_full_r", "formatted_feats_extract_l", "formatted_feats_extract_r"])
    
    return ddata


def validate_model(model,test_data,optimizer,criterion,args,combo):
    out_list = []
    model.eval()
    running_loss = 0.0
    loss_list = []

    if combo:
        test_set = get_dynamic_dataset_combo(test_data)
    else:
        test_set = get_dynamic_dataset(test_data)

    my_dataloader = DataLoader(test_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting validation...")
    for batch in tqdm(my_dataloader):
       
        if combo:
            correctness, feats_full_l, feats_full_r, feats_extract_l, feats_extract_r = batch
            feats_full_l = feats_full_l.data
            feats_full_r = feats_full_r.data
            feats_extract_l = feats_extract_l.data
            feats_extract_r = feats_extract_r.data
        else:
            correctness, feats_l, feats_r = batch
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
            feats_l = feats_l.to(device)
            feats_r = feats_r.to(device)
            
        target_scores = correctness.data.to(device)
        
        if combo:
            output_l,_ = model(feats_full_l.float(), feats_extract_l.float())
            output_r,_ = model(feats_full_r.float(), feats_extract_r.float())
        else:
            output_l,_ = model(feats_l)
            output_r,_ = model(feats_r)
            
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
    

def train_model(model,train_data,optimizer,criterion,args,combo=False,ex_data=None):
    model.train()
        
    running_loss = 0.0
    loss_list = []

    if combo:
        train_set = get_dynamic_dataset_combo(train_data)
    else:
        train_set = get_dynamic_dataset(train_data)

    my_dataloader = DataLoader(train_set,args.batch_size,collate_fn=sb.dataio.batch.PaddedBatch)
    print(f"batch_size: {args.batch_size}")
    print("starting training...")
    
    for batch in tqdm(my_dataloader, total=len(my_dataloader)):
        # batch = batch.to(device)
        if combo:
            correctness, feats_full_l, feats_full_r, feats_extract_l, feats_extract_r = batch
            feats_full_l = torch.nn.utils.rnn.pack_padded_sequence(
                feats_full_l.data, 
                (feats_full_l.lengths * feats_full_l.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            feats_full_r = torch.nn.utils.rnn.pack_padded_sequence(
                feats_full_r.data, 
                (feats_full_r.lengths * feats_full_r.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            feats_extract_l = torch.nn.utils.rnn.pack_padded_sequence(
                feats_extract_l.data, 
                (feats_extract_l.lengths * feats_extract_l.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            feats_extract_r = torch.nn.utils.rnn.pack_padded_sequence(
                feats_extract_r.data, 
                (feats_extract_r.lengths * feats_extract_r.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            feats_full_l = feats_full_l.to(device)
            feats_full_r = feats_full_r.to(device)
            feats_extract_l = feats_extract_l.to(device)
            feats_extract_r = feats_extract_r.to(device)
        else:
            correctness, feats_l, feats_r = batch
            # print(f"main feats_l.size: {feats_l.data.size()}")
            # print(f"lengths: {(feats_l.lengths * feats_l.data.size(1)).to(torch.int64)}")
            feats_l = torch.nn.utils.rnn.pack_padded_sequence(
                feats_l.data, 
                (feats_l.lengths * feats_l.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            # print(f"main 2 feats_l.size: {feats_l.data.size()}")
            feats_r = torch.nn.utils.rnn.pack_padded_sequence(
                feats_r.data, 
                (feats_r.lengths * feats_r.data.size(1)).to(torch.int64), 
                batch_first=True,
                enforce_sorted = False
            )
            feats_l = feats_l.to(device)
            feats_r = feats_r.to(device)
            # feats_l = feats_l
            # feats_r = feats_r
        #print("wavs:%s\n correctness:%s\n"%(wavs.data.shape,correctness))
        target_scores = correctness.data
        target_scores = target_scores.to(device)
        
        # #print(wavs_data.shape)
        # feats_l,feats_r = compute_feats(wavs_data,44100) 
    
        optimizer.zero_grad()
        if combo:
            output_l,_ = model(feats_full_l, feats_extract_l)
            output_r,_ = model(feats_full_r, feats_extract_r)
        else:
            output_l,_ = model(feats_l)
            output_r,_ = model(feats_r)
        loss_l = criterion(output_l,target_scores)
        loss_r = criterion(output_r,target_scores)
        loss = loss_l + loss_r

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        running_loss += loss.item()
        break
        
    
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


def main(args, config):
    #set up the torch objects
    print("creating model: %s"%args.model)
    torch.manual_seed(args.seed)
    N = args.N
    combo = False
    if args.feats == "XLSREncoder":
        feat_extractor = XLSREncoder_feats()
        dim_extractor = 512
        hidden_size = 512//2
        activation = nn.LeakyReLU
        att_pool_dim = 512

    elif args.feats == "XLSRFull":
        feat_extractor = XLSRFull_feats()
        dim_extractor = 1024
        hidden_size = 1024//2
        activation = nn.LeakyReLU
        att_pool_dim = 1024
        
    elif args.feats == "XLSRCombo":
        ## Tricky one! Check fat extractor
        combo = True
        feat_extractor = XLSRCombo_feats()
        dim_extractor = 1024
        hidden_size = 1024//2
        activation = nn.LeakyReLU
        
    elif args.feats == "HuBERTEncoder":
        feat_extractor = HuBERTEncoder_feats()
        dim_extractor = 512
        hidden_size = 512//2
        activation = nn.LeakyReLU
        att_pool_dim = 512
        
    elif args.feats == "HuBERTFull":
        feat_extractor = HuBERTFull_feats()
        dim_extractor = 768
        hidden_size = 768//2
        activation = nn.LeakyReLU
        att_pool_dim = 768
        
    elif args.feats == "Spec":  
        feat_extractor = Spec_feats()
        dim_extractor = 257
        hidden_size = 257//2
        activation = nn.LeakyReLU
        att_pool_dim = 256

    elif args.feats == "WhisperEncoder":  
        feat_extractor = WhisperEncoder_feats()
        dim_extractor = 768
        hidden_size = 768//2
        activation = nn.LeakyReLU
        att_pool_dim = 768
        
    else:
        print("Feats extractor not recognised")
        exit(1)

    if args.model == "LSTM":
        if combo:
            model = MetricPredictorLSTMCombo(dim_extractor, hidden_size, activation)
        else:
            model = MetricPredictorLSTM(dim_extractor, hidden_size, activation, att_pool_dim)
    elif args.model == "AttenPool":
        if combo:
            print("Combo Attention Pooling model not yet implemented")
            quit
        else:
            model = MetricPredictorAttenPool(att_pool_dim)
    else:
        print("Model not recognised")
        exit(1)

        # if args.exemplar:
        #     config["minerva_config"]["input_dim"] = 512
        #     model = ExemplarMetricPredictor(
        #         minerva_config = config["minerva_config"], 
        #         dim_extractor=512, 
        #         hidden_size=512//2, 
        #         activation=nn.LeakyReLU, 
        #         att_pool_dim = 512
        #     )


    feat_extractor.eval()
    feat_extractor.requires_grad_(False)
    #torchinfo.summary(model,(1,16000*8))
    #set up the model directory
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

    # if args.model == "XLSREncoder" and os.path.exists("data/xlsrencoder_feats.csv"):
    #     data = pd.read_csv("data/xlsrencoder_feats.csv")
    # else:

    data = pd.read_json(args.in_json_file)
    data["subset"] = "CEC1"
    if not args.use_CPC1:
        data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
        data2["subset"] = "CEC2"
        data = pd.concat([data, data2])
    data["predicted"] = np.nan  # Add column to store intel predictions

    feat_extractor.to(args.device)
    theset = "train_indep" if args.use_CPC1 else "train"
    data = extract_feats(feat_extractor, data, args, theset, combo)
    feat_extractor.to('cpu')
    data = get_disjoint_val_set(args, data)
    

    print(data[:50])

    # Split into train and val sets
    dis_val_data = data[data.validation == 7].copy()
    dis_lis_val_data = data[data.validation.isin([1, 3, 5, 7])].copy()
    dis_sys_val_data = data[data.validation.isin([2, 3, 6, 7])].copy()
    dis_scene_val_data = data[data.validation.isin([4, 5, 6, 7])].copy()
    train_data,val_data = train_test_split(data[data.validation == 0],test_size=0.1)
    #TODO set up validation to partition on scene
    if args.test_json_file is not None:
        test_data = pd.read_json(args.test_json_file)
        test_data["subset"] = "CEC1" if args.use_CPC1 else "CEC2"
        test_data["predicted"] = np.nan
        theset = "test_indep" if args.use_CPC1 else "test"
        feat_extractor.to(args.device)
        test_data = extract_feats(feat_extractor, test_data, args, theset, combo)
        feat_extractor.to('cpu')
        print(f"Using test data: {args.test_json_file}")
    else:
        test_data = dis_val_data
    print("Trainset: %s\nValset: %s\nDisValset: %s\nDisLisValset: %s\nDisSysValset: %s\nDisSceneValset: %s\nTestset: %s "%(train_data.shape[0],val_data.shape[0],dis_val_data.shape[0],dis_lis_val_data.shape[0],dis_sys_val_data.shape[0],dis_scene_val_data.shape[0],test_data.shape[0]))
    print("=====================================")

    # shuffle the training data
    if args.exemplar:
        ex_data = train_data.sample(n = config["minerva_config"]["num_ex"])
        ex_data = get_dynamic_dataset(ex_data)
        ex_dataloader = DataLoader(ex_data,config['minerva_config']['num_ex'],collate_fn=sb.dataio.batch.PaddedBatch)
        for batch_id, batch in enumerate(ex_dataloader):
            print(f"batch_id: {batch_id}")
            ex_targets, ex_feats_l, ex_feats_r = batch
            correctness = correctness.data
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
            # correctness = correctness.data
            # feats_l = feats_l.data
            # feats_r = feats_r.data

        print(f"correctness:\n{ex_targets}")
        print(f"feats_l:\n{ex_feats_l}")
        print(f"feats_r:\n{ex_feats_r}")

        print(f"correctness.size: {ex_targets.size()}, feat_l.size: {ex_feats_l.data.size()}, feats_r.size: {ex_feats_r.data.size()}")

        model.minerva.set_exemplars(
            ex_features_l = ex_feats_l,
            ex_features_r = ex_feats_r,
            ex_targets = ex_targets
        )

    else:
        ex_data = None

    train_data = train_data.sample(frac=1).reset_index(drop=True)

      
    # use this line for testing :) 
    #train_data = train_data[:50]
    #test_data = test_data[:50]
    #val_data = val_data[:50]

    model = model.to(args.device)

    if args.do_train:
        print("Starting training of model: %s\nlearning rate: %s\nseed: %s\nepochs: %s\nsave location: %s/"%(args.model,args.lr,args.seed,args.n_epochs,args.model_dir))
        print("=====================================")
        for epoch in range(args.n_epochs):


            model,optimizer,criterion,training_loss = train_model(model,train_data,optimizer,criterion,args,combo,ex_data)

            _,val_loss = validate_model(model,val_data,optimizer,criterion,args,combo)
            _,dis_val_loss = validate_model(model,dis_val_data,optimizer,criterion,args,combo)
            _,dis_lis_val_loss = validate_model(model,dis_lis_val_data,optimizer,criterion,args,combo)
            _,dis_sys_val_loss = validate_model(model,dis_sys_val_data,optimizer,criterion,args,combo)
            _,dis_scene_val_loss = validate_model(model,dis_scene_val_data,optimizer,criterion,args,combo)
            if args.test_json_file is not None:
                _,eval_loss = validate_model(model,test_data,optimizer,criterion,args,combo)

            if not args.skip_wandb:
                log_dict = {
                    "val_rmse": val_loss**0.5,
                    "dis_val_rmse": dis_val_loss**0.5,
                    "dis_lis_val_rmse": dis_lis_val_loss**0.5,
                    "dis_sys_val_rmse": dis_sys_val_loss**0.5,
                    "dis_scene_val_rmse": dis_scene_val_loss**0.5,
                    "train_rmse": training_loss**0.5
                }
                if args.test_json_file is not None:
                    log_dict["eval_rmse"] = eval_loss**0.5
                
                wandb.log(log_dict)
            
            save_model(model,optimizer,epoch,args,val_loss)
            torch.cuda.empty_cache()
            print("Epoch: %s"%(epoch))
            print("\tTraining Loss: %s"%(training_loss))
            print("\tValidation Loss: %s"%(val_loss))
            print("\tDisjoint validation Loss: %s"%(dis_val_loss))
            print("\tDisjoint listener validation Loss: %s"%(dis_lis_val_loss))
            print("\tDisjoint system validation Loss: %s"%(dis_sys_val_loss))
            print("\tDisjoint scene validation Loss: %s"%(dis_scene_val_loss))
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
    val_predictions,val_loss = validate_model(model,val_data,optimizer,criterion,args,combo)
    val_error = val_loss**0.5
    val_p_corr = np.corrcoef(np.array(val_predictions),val_data["correctness"])[0][1]
    val_s_corr = spearmanr(np.array(val_predictions),val_data["correctness"])[0]
    val_std = std_err(np.array(val_predictions), val_data["correctness"])
    val_data["predicted"] = val_predictions
    val_data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file + "_val_preds.csv", index=False)

    dis_val_predictions,dis_val_loss = validate_model(model,dis_val_data,optimizer,criterion,args,combo)
    dis_val_error = dis_val_loss**0.5
    dis_val_p_corr = np.corrcoef(np.array(dis_val_predictions),dis_val_data["correctness"])[0][1]
    dis_val_s_corr = spearmanr(np.array(dis_val_predictions),dis_val_data["correctness"])[0]
    dis_val_std = std_err(np.array(dis_val_predictions), dis_val_data["correctness"])
    dis_val_data["predicted"] = dis_val_predictions
    dis_val_data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file + "_dis_val_preds.csv", index=False)

    dis_lis_val_predictions,dis_lis_val_loss = validate_model(model,dis_lis_val_data,optimizer,criterion,args,combo)
    dis_lis_val_error = dis_lis_val_loss**0.5
    dis_lis_val_p_corr = np.corrcoef(np.array(dis_lis_val_predictions),dis_lis_val_data["correctness"])[0][1]
    dis_lis_val_s_corr = spearmanr(np.array(dis_lis_val_predictions),dis_lis_val_data["correctness"])[0]
    dis_lis_val_std = std_err(np.array(dis_lis_val_predictions), dis_lis_val_data["correctness"])
    dis_lis_val_data["predicted"] = dis_lis_val_predictions
    dis_lis_val_data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file + "_dis_lis_val_preds.csv", index=False)

    dis_sys_val_predictions,dis_sys_val_loss = validate_model(model,dis_sys_val_data,optimizer,criterion,args,combo)
    dis_sys_val_error = dis_sys_val_loss**0.5
    dis_sys_val_p_corr = np.corrcoef(np.array(dis_sys_val_predictions),dis_sys_val_data["correctness"])[0][1]
    dis_sys_val_s_corr = spearmanr(np.array(dis_sys_val_predictions),dis_sys_val_data["correctness"])[0]
    dis_sys_val_std = std_err(np.array(dis_sys_val_predictions), dis_sys_val_data["correctness"])
    dis_sys_val_data["predicted"] = dis_sys_val_predictions
    dis_sys_val_data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file + "_dis_sys_val_preds.csv", index=False)

    dis_scene_val_predictions,dis_scene_val_loss = validate_model(model,dis_scene_val_data,optimizer,criterion,args,combo)
    dis_scene_val_error = dis_scene_val_loss**0.5
    dis_scene_val_p_corr = np.corrcoef(np.array(dis_scene_val_predictions),dis_scene_val_data["correctness"])[0][1]
    dis_scene_val_s_corr = spearmanr(np.array(dis_scene_val_predictions),dis_scene_val_data["correctness"])[0]
    dis_scene_val_std = std_err(np.array(dis_scene_val_predictions), dis_scene_val_data["correctness"])
    dis_scene_val_data["predicted"] = dis_scene_val_predictions
    dis_scene_val_data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file + "_dis_scene_val_preds.csv", index=False)

    
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
        predictions,test_loss = validate_model(model,test_data,optimizer,criterion,args,combo)
        predictions_fitted = np.asarray(predictions)/100
        #apply the logistic mapping
        predictions_fitted = logit_func(predictions_fitted,a,b)
        test_data["predicted"] = predictions
        test_data["predicted_fitted"] = predictions_fitted*100
        test_data[["scene", "listener", "system", "predicted","predicted_fitted"]].to_csv(args.out_csv_file + "_test_preds.csv", index=False)
        error = test_loss**0.5 * 100
        p_corr = np.corrcoef(np.array(predictions),test_data["correctness"])[0][1]
        s_corr = spearmanr(np.array(predictions),test_data["correctness"])[0]
        std = std_err(np.array(predictions), test_data["correctness"])
        error_fitted = rmse_score(np.array(predictions_fitted)*100,test_data["correctness"])
        p_corr_fitted = np.corrcoef(np.array(predictions_fitted)*100,test_data["correctness"])[0][1]
        s_corr_fitted = spearmanr(np.array(predictions_fitted)*100,test_data["correctness"])[0]
        std_fitted = std_err(np.array(predictions_fitted)*100,test_data["correctness"])

        print("Test Loss: %s"%test_loss)
    # else:
    #     print("Testing model on train+val set")
    #     predictions,test_loss = validate_model(model,data,optimizer,criterion,args,combo)
    #     data["predicted"] = predictions
    #     data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file, index=False)
    
    with open (args.summ_file, "a") as f:
        csv_writer = csv.writer(f)
        if args.test_json_file is not None:
            csv_writer.writerow([args.out_csv_file.split("/")[-1].strip(".csv"), val_error, val_p_corr, val_s_corr, val_std, error, std, p_corr, s_corr, error_fitted, std_fitted, p_corr_fitted, s_corr_fitted])
        else:
            csv_writer.writerow([
                args.out_csv_file.split("/")[-1].strip(".csv"), 
                val_error, val_p_corr, val_s_corr, val_std, 
                dis_val_error, dis_val_p_corr, dis_val_s_corr, dis_val_std, 
                dis_lis_val_error, dis_lis_val_p_corr, dis_lis_val_s_corr, dis_lis_val_std, 
                dis_sys_val_error, dis_sys_val_p_corr, dis_sys_val_s_corr, dis_sys_val_std, 
                dis_scene_val_error, dis_scene_val_p_corr, dis_scene_val_s_corr, dis_scene_val_std
            ])

    
    print("=====================================")

    
    if not args.skip_wandb:
        run.finish()

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
