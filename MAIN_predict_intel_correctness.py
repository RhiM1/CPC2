
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




DATAROOT = "data/" 
df_listener = pd.read_json("data/clarity_data/metadata/listeners.json")

resampler = T.Resample(32000, 16000).to("cuda:0")

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


def get_mean(scores):
    out_list = []
    for el in scores:
        el = el.strip("[").strip("]").split(",")
        el = [float(a) for a in el]
        #print(el,type(el))
        out_list.append(el)
    return torch.Tensor(out_list).unsqueeze(0)

def validate_model(model,test_data,optimizer,criterion,N):
    out_list = []
    model.eval()
    name_list = test_data["signal"]
    correctness_list = test_data["correctness"]
    listener_list = test_data["listener"]
    scene_list = test_data["scene"]
    subset_list = test_data["subset"]

    #haspi_list = test_data["haspi"]
    #print(name_list)
    running_loss = 0.0
    loss_list = []
    test_dict = {}
    for sub,name,corr,scene,lis in  zip(subset_list,name_list,correctness_list,scene_list,listener_list):
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
    test_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
    #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
    test_set.set_output_keys(["wav", "formatted_correctness"])

    my_dataloader = DataLoader(test_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting validation...")
    for batch in tqdm(my_dataloader):
        batch = batch.to("cuda:0")
        wavs,correctness = batch
        correctness =correctness.data
        wavs_data = wavs.data
        #print("wavs:%s\n correctness:%s\naudiogram:%s"%(wavs.data.shape,correctness,audiogram))
        target_scores = correctness
        #feats = wavs_data
        feats_l,feats_r = compute_feats(wavs_data,44100) 
        #print(feats.shape)
       
        output_l,_ = model(feats_l.float())
        output_r,_ = model(feats_r.float())
    
        output = max(output_l,output_r)
        loss = criterion(output,target_scores)
        #for x1,y1 in zip(output.detach().cpu().numpy(),target_scores.cpu().detach().numpy()):
        #    print("P: %s | T: %s"%(x1,y1))
        out_list.append(output.detach().cpu().numpy()[0][0]*100)

        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item()
    #print("Average MSE loss: %s"%(sum(loss_list)/len(loss_list)))

    return out_list,sum(loss_list)/len(loss_list)

def test_model(model,test_data,optimizer,criterion,N):
    out_list = []
    model.eval()
    name_list = test_data["signal"]
    correctness_list = test_data["correctness"]
    listener_list = test_data["listener"]
    scene_list = test_data["scene"]
    #haspi_list = test_data["haspi"]
    #print(name_list)
    running_loss = 0.0
    loss_list = []
    test_dict = {}
    for name,corr,scene,lis in  zip(name_list,correctness_list,scene_list,listener_list,):
        test_dict[name] = {"signal": name,"correctness":corr,"scene": scene,"listener":lis}

    dynamic_items = [
        {"func": lambda l: format_correctness(l),
        "takes": "correctness",
        "provides": "formatted_correctness"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/train/%s.wav"%(DATAROOT,l),32000),
        #"takes": "signal",
        #"provides": "wav"},
        {"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/test.%s/%s.wav"%(DATAROOT,N,l),44100),
        "takes": "signal",
        "provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/scenes//%s_target_anechoic.wav"%(DATAROOT,l),44100),
        #"takes": "scene",
        #"provides": "clean_wav"},
    ]
    test_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
    #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
    test_set.set_output_keys(["wav", "formatted_correctness"])

    my_dataloader = DataLoader(test_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting testing...")
    for batch in tqdm(my_dataloader):
        batch = batch.to("cuda:0")
        wavs,correctness = batch
        correctness =correctness.data
        wavs_data = wavs.data
        #print("wavs:%s\n correctness:%s\naudiogram:%s"%(wavs.data.shape,correctness,audiogram))
        target_scores = correctness
        #feats = wavs_data
        feats_l,feats_r = compute_feats(wavs_data,44100) 
        #print(feats.shape)
       
        output_l,_ = model(feats_l.float())
        output_r,_ = model(feats_r.float())
      
        output = max(output_l,output_r)
        loss = criterion(output,target_scores)

        #for x1,y1 in zip(output.detach().cpu().numpy(),target_scores.cpu().detach().numpy()):
        #    print("P: %s | T: %s"%(x1,y1))
        out_list.append(output.detach().cpu().numpy()[0][0]*100)

        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item()
    #print("Average MSE loss: %s"%(sum(loss_list)/len(loss_list)))

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
    



def train_model(model,train_data,optimizer,criterion,N):
    model.train()
    name_list = train_data["signal"]
    correctness_list = train_data["correctness"]
    #haspi_list = train_data["HASPI"]
    scene_list = train_data["scene"]
    listener_list = train_data["listener"]
    subset_list = train_data["subset"]
    #print(name_list)
    #columns_titles = ["signal",'scene', 'listener', 'system', 'mbstoi', 'correctness', 'predicted']
    #train_data = train_data.reindex(columns_titles)
    #train_data = train_data.to_dict()
    running_loss = 0.0
    loss_list = []
   
    train_dict = {}
    for sub,name,corr,scene,lis in  zip(subset_list,name_list,correctness_list,scene_list,listener_list):
        train_dict[name] = {"subset":sub,"signal": name,"correctness":corr,"scene": scene,"listener":lis}





    dynamic_items = [
        {"func": lambda l: format_correctness(l),
        "takes": "correctness",
        "provides": "formatted_correctness"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/train/%s.wav"%(DATAROOT,l),32000),
        #"takes": "signal",
        #"provides": "wav"},
        {"func": lambda l,y: audio_pipeline("%s/clarity_data/HA_outputs/train.%s/%s/%s.wav"%(DATAROOT,N,y,l),32000),
        "takes": ["signal","subset"],
        "provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/scenes//%s_target_anechoic.wav"%(DATAROOT,l),44100),
        #"takes": "scene",
        #"provides": "clean_wav"},
        #{"func": lambda l: convert_audiogram(l),
        #"takes": "listener",
        #"provides": "audiogram_np"}
    ]
    train_set = sb.dataio.dataset.DynamicItemDataset(train_dict,dynamic_items)
    #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
    train_set.set_output_keys(["wav", "formatted_correctness"])

    my_dataloader = DataLoader(train_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting training...")
    
    for batch in tqdm(my_dataloader, total=len(my_dataloader)):
        batch = batch.to("cuda:0")
        wavs,correctness = batch
        correctness =correctness.data
        wavs_data = wavs.data
        #print("wavs:%s\n correctness:%s\n"%(wavs.data.shape,correctness))
        target_scores = correctness
        
        #print(wavs_data.shape)
        feats_l,feats_r = compute_feats(wavs_data,44100) 
        
    
        optimizer.zero_grad()
        output_l,_ = model(feats_l.float())
        output_r,_ = model(feats_r.float())
        loss_l = criterion(output_l,target_scores)
        loss_r = criterion(output_r,target_scores)
        loss = loss_l + loss_r

        loss.backward()
        optimizer.step()
        loss_list.append(loss)
        running_loss += loss.item()
    #print("Average Training loss: %s"%(sum(loss_list)/len(loss_list)))
    
    return model,optimizer,criterion,sum(loss_list)/len(loss_list)


def convert_audiogram(listener):
    audiogram_l =  format_audiogram(np.array(df_listener[listener]["audiogram_levels_l"]))
    audiogram_r =  format_audiogram(np.array(df_listener[listener]["audiogram_levels_r"]))
    audiogram = [audiogram_l,audiogram_r]
    return audiogram



def save_model(model,opt,epoch,args,val_loss):
    p = args.model_dir
    if not os.path.exists(p):
        os.mkdir(p)
    m_name = "%s-%s"%(args.model,args.seed)
    torch.save(model.state_dict(),"%s/%s_%s_%s_model.pt"%(p,m_name,epoch,val_loss))
    torch.save(opt.state_dict(),"%s/%s_%s_%s_opt.pt"%(p,m_name,epoch,val_loss))



def main(args):
    #set up the torch objects
    print("creating model: %s"%args.model)
    torch.manual_seed(args.seed)
    N = args.N
    if args.model == "XLSREncoder":
        from models.ni_predictors import XLSRMetricPredictorEncoder
        model = XLSRMetricPredictorEncoder().to("cuda:0")
    elif args.model == "XLSRFull":
        from models.ni_predictors import XLSRMetricPredictorFull
        model = XLSRMetricPredictorFull().to("cuda:0")
    elif args.model == "XLSRCombo":
        from models.ni_predictors import XLSRMetricPredictorCombo
        model = XLSRMetricPredictorCombo().to("cuda:0")
    elif args.model == "HuBERTEncoder":
        from models.ni_predictors import HuBERTMetricPredictorEncoder
        model = HuBERTMetricPredictorEncoder().to("cuda:0")
    elif args.model == "HuBERTFull":
        from models.ni_predictors import HuBERTMetricPredictorFull
        model = HuBERTMetricPredictorFull().to("cuda:0")
    elif args.model == "Spec":  
        from models.ni_predictors import SpecMetricPredictor
        model = SpecMetricPredictor().to("cuda:0")
    else:
        print("Model not recognised")
        exit(1)
    model = model.to("cuda:0")
    #torchinfo.summary(model,(1,16000*8))
    #set up the model directory
    today = datetime.date.today()
    date = today.strftime("%H-%M-%d-%b-%Y")

    if "indep" in args.in_json_file:
        model_dir = "save/CPC2_train%s_%s_%s_%s_indep"%(args.N,args.model,date,args.seed)
    else:
        model_dir = "save/CPC2_train%s_%s_%s_%s"%(args.N,args.model,date,args.seed)
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


            model,optimizer,criterion,training_loss = train_model(model,train_data,optimizer,criterion,args.N)

            _,val_loss = validate_model(model,val_data,optimizer,criterion,args.N)
            
            save_model(model,optimizer,epoch,args,val_loss)
            torch.cuda.empty_cache()
            print("Epoch: %s | Training Loss: %s | Validation Loss: %s"%(epoch,training_loss.item(),val_loss))
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
    val_predictions,_ = validate_model(model,val_data,optimizer,criterion,args.N)
    
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
        predictions,test_loss = validate_model(model,data,optimizer,criterion,args.N)
    
        data["predicted"] = predictions
        data[["scene", "listener", "system", "predicted"]].to_csv(args.out_csv_file, index=False)
    print("Test Loss: %s"%test_loss)
    print("=====================================")

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
    args = parser.parse_args()
    main(args)
