
import sys
sys.settrace 
import torch
from speechbrain.nnet.losses import mse_loss 
import speechbrain as sb

from speechbrain.processing.features import spectral_magnitude,STFT
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from huBERT_wrapper import HuBERTWrapper_extractor
import torchaudio.transforms as T
import pandas as pd
import tqdm


data_root= "/jmain02/home/J2AD003/txk68/gxc35-txk68/data/clarity_CPC1_data/clarity_data/"
json_path =  "/jmain02/home/J2AD003/txk68/gxc35-txk68/data/clarity_CPC1_data/metadata/CPC1.train.json"
def compute_feats(wavs):
    """Feature computation pipeline"""
    stft = STFT(hop_length=16,win_length=32,sample_rate=16000,n_fft=512,window_fn=torch.hamming_window)
    feats = stft(wavs)
    feats = spectral_magnitude(feats, power=0.5)
    feats = torch.log1p(feats)
    return feats 

def dataio_prep(json_path,data_root):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipelines
    @sb.utils.data_pipeline.takes("signal")
    @sb.utils.data_pipeline.provides("noisy_sig")
    def noisy_pipeline(noisy_wav):
        in_wav = sb.dataio.dataio.read_audio(data_root+"/HA_outputs/train/"+noisy_wav+".wav")
        print(in_wav.shape)
        resample_rate = 16000
        #sum stero channels 
        in_wav = in_wav[:,0] + in_wav[:,1]
        #resample to 16000 for input to HASPI
        resampler = T.Resample(32000, resample_rate, dtype=in_wav.dtype)
        resampled_waveform = resampler(in_wav)

        return resampled_waveform

    @sb.utils.data_pipeline.takes("scene")
    @sb.utils.data_pipeline.provides("clean_sig")
    def clean_pipeline(clean_wav):
        in_wav = sb.dataio.dataio.read_audio(data_root+"/scenes/"+clean_wav +"_target_anechoic.wav")
        resample_rate = 16000
        #sum stero channels 
        in_wav = in_wav[:,0] + in_wav[:,1]
        #resample to 16000 for input to HASPI
        resampler = T.Resample(44100, resample_rate, dtype=in_wav.dtype)
        resampled_waveform = resampler(in_wav)

        return resampled_waveform

    train_data = pd.read_json(json_path)
    name_list = train_data["signal"]
    correctness_list = train_data["correctness"]
    scene_list = train_data["scene"]
    train_dict = {}
    for name,corr,scene, in  zip(name_list,correctness_list,scene_list):
        train_dict[name] = {"signal": name,"correctness":corr,"scene": scene}
    # Define datasets
    dataset = {}
    
    dataset = sb.dataio.dataset.DynamicItemDataset(
        train_dict,
        dynamic_items=[noisy_pipeline, clean_pipeline],
        output_keys=["id", "noisy_sig", "clean_sig","correctness"],
    )

    return dataset
    

dataset = dataio_prep(json_path,data_root)
    

mod_e = HuBERTWrapper_extractor().to("cuda")
mod_o = HuBERTWrapper_full().to("cuda")
print(dataset)
dataloader = SaveableDataLoader(dataset, collate_fn=PaddedBatch,
    batch_size=1)

out_list = []
print("STARTING!")
for batch in tqdm.tqdm(dataloader):
    #print(batch.id)
    #batch.to("cuda")
    id = batch.id
    clean_sig,lens_c = batch.clean_sig
    noisy_sig,lens_n = batch.noisy_sig
    print(lens_c,lens_n)
    #print(clean_sig.shape)
    clean_rep_e = mod_e(clean_sig).permute(0,2,1)
    noisy_rep_e = mod_e(noisy_sig).permute(0,2,1)
    clean_rep_o = mod_o(clean_sig)
    noisy_rep_o = mod_o(noisy_sig)
    print(clean_rep_e.shape,noisy_rep_e.shape)
    print(clean_rep_o.shape,noisy_rep_o.shape)
    extract_feats_mse = mse_loss(clean_rep_e,noisy_rep_e,length=lens_c,reduction="batch").detach().cpu().numpy()
    hidden_state_mse = mse_loss(clean_rep_o,noisy_rep_o,length=lens_c,reduction="batch").detach().cpu().numpy()
    spec_mse = mse_loss(compute_feats(clean_sig),compute_feats(noisy_sig),length=lens_c,reduction="batch")
    #print(extract_feats_mse,hidden_state_mse,)
    correctness = batch.correctness
    for i,ef,hs,sg,c in zip(id,extract_feats_mse,hidden_state_mse,spec_mse,correctness):
        print("%s,%s,%s,%s,%s\n"%(i,ef,hs,sg,c.item()))
        out_list.append("%s,%s,%s,%s,%s\n"%(i,ef,hs,sg,c.item()))
    
with open("CPC1_train_hubert_lens3.csv","w") as f:
    f.write("name,ef_mse,ol_mse,sg_mse\n")
    f.writelines(out_list)