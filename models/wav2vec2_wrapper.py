from torch import Tensor, nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder
#import librosa 


def get_audio(file, new_sr):
    # Source:
    # https://huggingface.co/docs/datasets/_modules/datasets/features/audio.html#Audio

    array, sampling_rate = sf.read(file)
    array = array.T
    array = librosa.to_mono(array)
    if new_sr and new_sr != sampling_rate:
        array = librosa.resample(
            array,
            orig_sr=sampling_rate,
            target_sr=new_sr,
            res_type="kaiser_best"
        )
        sampling_rate = new_sr
    return array, sampling_rate


class Wav2Vec2Wrapper(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sampling rate.
        self.sampling_rate = 16_000

        # Construct preprocessing helper.
        self.helper = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        # Create model.
        self.model = Wav2Vec2Model.from_pretrained(
            f"models/facebook/wav2vec2-xls-r-300m")

    def forward(self, data: Tensor):
        # Model.
        inputs = self.helper(
            data,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        #print(inputs)
        input = inputs["input_values"].to(data.device)
        return self.model(input)



class Wav2Vec2Wrapper_no_helper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Wav2Vec2Model.from_pretrained(
            f"facebook/wav2vec2-xls-r-300m")


    def forward(self, data: Tensor):
        
        return self.model(data)
        
class Wav2Vec2Wrapper_encoder_only(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Wav2Vec2Model.from_pretrained(
            f"models/facebook/wav2vec2-xls-r-300m").feature_extractor
        

    def forward(self, data: Tensor):
        
        return self.model(data)

if __name__ == "__main__":
    import soundfile as sf
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from speechbrain.nnet.losses import mse_loss 
    import os
    import csv
    from speechbrain.processing.features import spectral_magnitude,STFT
    mod = Wav2Vec2Wrapper()
    try:
        from models.will_utils import channel_sort,normalize
    except:
        from will_utils import channel_sort,normalize

    def compute_feats(wavs):
        """Feature computation pipeline"""
        stft = STFT(hop_length=16,win_length=32,sample_rate=16000,n_fft=512,window_fn=torch.hamming_window)
        feats = stft(torch.from_numpy(wavs).unsqueeze(0))
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats 


    
    
    clean_root= "/fastdata/acp20glc/VoiceBank/clean_testset_wav_16k/"
    noisy_root = "/fastdata/acp20glc/VoiceBank/noisy_testset_wav_16k/"
    
    
    
    f_names = os.listdir(clean_root)
    with open("/fastdata/acp20glc/projects/MetricGANdiscrim_feats/log_testset.csv") as f:
        csv_reader = csv.reader(f)

    
        

        hs_list = []
        ef_list = []
        next(csv_reader,None)
        for row in csv_reader:
            clean_wav_ref,fs = sf.read(clean_root+"p257_001.wav")
            file, loc, snr = row
            clean_wav,fs = sf.read(clean_root+file+".wav")
            noisy_wav,fs = sf.read(noisy_root+file+".wav")
            #noisy_wav = np.random.random(clean_wav.shape)
            """
            if len(clean_wav) > len(clean_wav_ref):
                clean_wav = clean_wav[:len(clean_wav_ref)]
            else:
                clean_wav_ref = clean_wav_ref[:len(clean_wav)]
            """
            #noisy_wav = clean_wav_ref
            clean_rep = mod(torch.from_numpy(clean_wav))

            noisy_rep = mod(torch.from_numpy(noisy_wav))
            hidden_state_mse = mse_loss(clean_rep["last_hidden_state"],noisy_rep["last_hidden_state"]).detach().cpu().numpy()
            extract_feats_mse = mse_loss(clean_rep["extract_features"],noisy_rep["extract_features"]).detach().cpu().numpy()
            hidden_state_cosine = np.mean(torch.nn.CosineSimilarity()(clean_rep["last_hidden_state"],noisy_rep["last_hidden_state"]).detach().cpu().numpy())
            extract_feats_cosine = np.mean(torch.nn.CosineSimilarity()(clean_rep["extract_features"],noisy_rep["extract_features"]).detach().cpu().numpy())

            ###################################################################
            fig, ax = plt.subplots(2,2,figsize=(3.378,3))
            
             
            ax[0,0].imshow(compute_feats(clean_wav).T.flipud().squeeze(0),aspect='auto')
            #plt.xlabel(r'$\mathbf{S}_\mathrm{SG}$')
            ax[0,0].set_xlabel(r'$T$')
            ax[0,0].set_ylabel(r'$F_{Hz}$')
            ax[0,0].set_title(r'$\mathbf{S}_\mathrm{SG}$')
            ax[0,1].imshow(compute_feats(noisy_wav).T.flipud().squeeze(0),aspect='auto')
            ax[0,1].set_xlabel(r'$\mathbf{S}_\mathrm{SG}$')
            ax[0,1].set_xlabel(r'$T$')
            ax[0,1].set_ylabel(r'$F_{Hz}$')
            ax[0,1].set_title(r'$\mathbf{X}_\mathrm{SG}$')
            
            clean_rep_disp = normalize(clean_rep["extract_features"].T.squeeze().detach().cpu().numpy())
            print(clean_rep_disp.shape)
            chan_idx_clean = channel_sort(clean_rep_disp)
            print(chan_idx_clean.shape)
            clean_rep_disp = clean_rep_disp[chan_idx_clean,:]
            print(clean_rep_disp.shape)

            noisy_rep_disp = normalize(noisy_rep["extract_features"].T.squeeze().detach().cpu().numpy())
            noisy_rep_disp = noisy_rep_disp[chan_idx_clean,:]
            print(clean_rep_disp.shape)
            ax[1,0].imshow(np.flipud(clean_rep_disp),cmap="plasma",aspect='auto')

            ax[1,0].set_ylabel(r'$F$')
            ax[1,0].set_xlabel(r'$T$')
            ax[1,0].set_title(r'XLSR $\mathbf{S}_\mathrm{FE}$')

            #plt.subplot(144)
            ax[1,1].imshow(np.flipud(noisy_rep_disp),cmap="plasma",aspect='auto')
            ax[1,1].set_ylabel(r'$F$')
            ax[1,1].set_xlabel(r'$T$')
            ax[1,1].set_title(r'XLSR $\mathbf{X}_\mathrm{FE}$')

       
            plt.tight_layout()
            
            plt.savefig(file+"_xlsr.png")
            plt.show()
            #plt.close()
            #input(">>>")           

        print(np.mean(hs_list),np.var(hs_list))
        print(np.mean(ef_list),np.var(ef_list))
    """
    for wav in ["p232_025.wav","p232_020.wav"]:
        audio_data,fs = get_audio(wav,16000)
        audio_data_np = np.float32(audio_data)
        audio_data_pt = torch.from_numpy(audio_data_np)
        #print(mod)
        print(audio_data_pt.shape)
        rep = mod(audio_data_pt)
        #print(rep.shape)
        #print(rep)
        print(rep["last_hidden_state"],rep["last_hidden_state"].shape)
        print(rep["extract_features"],rep["extract_features"].shape)

        hidden_state = rep["last_hidden_state"]
        feats = rep["extract_features"]
        
        fig, axs = plt.subplots(3)
        axs[0].plot(audio_data_np)
        axs[1].imshow(feats.squeeze().detach().numpy())
        axs[2].imshow(hidden_state.squeeze().detach().numpy())

        plt.savefig("out_%s.png"%wav.split(".")[0])
    """
    