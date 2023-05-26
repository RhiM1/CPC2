
import torch
from speechbrain.nnet.losses import mse_loss 
import speechbrain as sb
from speechbrain.processing.features import spectral_magnitude,STFT
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
import torchaudio.transforms as T
import pandas as pd
import tqdm
from models.huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
mod_fe = HuBERTWrapper_extractor().to("cuda")
mod_full = HuBERTWrapper_full().to("cuda")

resampler = T.Resample(44100, 16000)
resampler2 = T.Resample(32000, 16000)
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
        #sum stero channels 
        in_wav = in_wav[:,0]
        #resample to 16000 for input to HASPI
        
        resampled_waveform = resampler2(in_wav)

        return resampled_waveform

    @sb.utils.data_pipeline.takes("scene")
    @sb.utils.data_pipeline.provides("clean_sig")
    def clean_pipeline(clean_wav):
        in_wav = sb.dataio.dataio.read_audio(data_root+"/scenes/"+clean_wav +"_target_anechoic.wav")
        print(in_wav.shape)

        resample_rate = 16000
        #sum stero channels 
        in_wav = in_wav[:,0]
        #resample to 16000 for input to HASPI
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

print(len(dataset))
dataloader = SaveableDataLoader(dataset, collate_fn=PaddedBatch,
    batch_size=1)

out_list = []
for batch in tqdm.tqdm(dataloader):
    #print(batch.id)
    id = batch.id
    print(id)
    clean_sig,lens = batch.clean_sig
    noisy_sig,lens = batch.noisy_sig
    print(clean_sig.shape,noisy_sig.shape)
    longest = max(clean_sig.shape[1],noisy_sig.shape[1])
    clean_sig = torch.nn.functional.pad(clean_sig,(0,longest-clean_sig.shape[1]))
    noisy_sig = torch.nn.functional.pad(noisy_sig,(0,longest-noisy_sig.shape[1]))
    print(clean_sig.shape,noisy_sig.shape)
    print("-----")

    clean_sig = clean_sig.to("cuda:0")
    noisy_sig = noisy_sig.to("cuda:0")
    clean_rep_fe = mod_fe(clean_sig.to("cuda:0"))
    noisy_rep_fe = mod_fe(noisy_sig.to("cuda:0"))
    clean_rep = mod_full(clean_sig.to("cuda:0")).swapaxes(0,1)
    noisy_rep = mod_full(noisy_sig.to("cuda:0")).swapaxes(0,1)

    print(noisy_rep_fe.shape)
    print(noisy_rep_fe.shape)
    print(clean_rep.shape)
    print(noisy_rep.shape)
    hidden_state_mse = mse_loss(clean_rep.cpu(),noisy_rep.cpu(),lens,reduction="batch").detach().cpu().numpy()
    extract_feats_mse = mse_loss(noisy_rep_fe.cpu(),clean_rep_fe.cpu(),lens,reduction="batch").detach().cpu().numpy()
    spec_mse = mse_loss(compute_feats(clean_sig).cpu(),compute_feats(noisy_sig).cpu(),lens,reduction="batch")
    #print(extract_feats_mse,hidden_state_mse,)
    correctness = batch.correctness
    for i,ef,hs,sg,c in zip(id,extract_feats_mse,hidden_state_mse,spec_mse,correctness):
        print("%s,%s,%s,%s,%s\n"%(i,ef,hs,sg.item(),c.item()))
        out_list.append("%s,%s,%s,%s,%s\n"%(i,ef,hs,sg.item(),c.item()))
    
with open("CPC1_train_hubert2.csv","w") as f:
    f.write("name,ef_mse,ol_mse,sg_mse,correctness\n")
    f.writelines(out_list)