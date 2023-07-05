from torch import Tensor, nn
# import fairseq
import torch
import torch.nn.functional as F
from torchaudio.transforms import SlidingWindowCmn
# import speechbrain as sb
from transformers import HubertModel, HubertConfig, WhisperConfig, WhisperModel, WhisperFeatureExtractor, WhisperForConditionalGeneration


class HuBERTWrapper_extractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"
        # models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        # self.model = models[0].feature_extractor
        # self.model.requires_grad_(False)

        configuration = HubertConfig()
        model = HubertModel(configuration)
        self.model = model.feature_extractor
        print(self.model)


        
    def forward(self, data: Tensor):
        #print(self.model)
        #print(data.shape)
        return self.model(data)


class HuBERTWrapper_full(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"

        # models = fairseq.checkpoint_utils.load_model_ensemble([ckpt_path])
        # full_model = models[0][0]
        # full_model.features_only =True
        # self.model = full_model
        
        configuration = HubertConfig()
        self.model = HubertModel(configuration)
        print(self.model)
        
    

    def forward(self, data: Tensor):
        
        """
        my_output = None
        def my_hook(module_,input_,output_):
            nonlocal my_output
            my_output = output_

        a_hook = self.model.encoder.layers[6].final_layer_norm.register_forward_hook(my_hook)
        self.model(data)
        a_hook.remove()
        """
        
        my_output =self.model(data)
        # print(my_output)
        # quit()
        # return my_output
        return my_output[0]
    
class WhisperWrapper_encoder(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_feat_extractor = use_feat_extractor
        self.layer = layer

        if use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            model = WhisperModel.from_pretrained("openai/whisper-small")
        else:
            model = WhisperModel.from_pretrained(pretrained_model)

        self.model = model.encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # print(self.model)

        
    def forward(self, data):

        # print(f"wrapper input_features.size: {data['input_features'].size()}")
        # print(f"wrapper attention_mask.size: {data['attention_mask'].size()}")
        if self.use_feat_extractor:
            data = self.feature_extractor(data[0].to('cpu'), sampling_rate = 16000, return_tensors = 'pt')
            data = data.input_features.to(self.device)

        # print(f"wrapper data:\n{data.size()}")

        if self.layer is None:
            data = self.model(
                input_features = data, 
                # attention_mask = data['attention_mask'],
                return_dict = True
            )
            data = data[0]
        else:
            data = self.model(
                input_features = data, 
                # attention_mask = data['attention_mask'],
                return_dict = True,
                output_hidden_states = True
            )
            data = data.hidden_states[self.layer]
            # print(data)

            # for layerID, layer in enumerate(data):
            #     print(layerID, layer.size())
            # quit()
            # print(data)
            # print(data["hidden_states"])
            # print(f"\nnum hidden states:\n{len(data['hidden_states'])}\n")
            # print(f"\nhidden state 0 size:\n{data['hidden_states'][0].size()}\n")
            # quit()
            # print(data)
            # print(data["hidden_states"])
            # print(f"\nnum hidden states:\n{len(data['hidden_states'])}\n")
            # print(f"\nhidden state 0 size:\n{data['hidden_states'][0].size()}\n")
            # quit()
        # data = self.model(data)
        return data
    
    
class WhisperWrapper_full(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_feat_extractor = use_feat_extractor
        if layer is None:
            self.layer = 12
        else:
            self.layer = layer

        if use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            # self.model = WhisperModel.from_pretrained("openai/whisper-small")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        else:
            self.model = WhisperModel.from_pretrained(pretrained_model)
            # self.model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # print(self.model)    
    
        
    def forward(self, data):

        # print(f"wrapper input_features.size: {data['input_features'].size()}")
        # print(f"wrapper attention_mask.size: {data['attention_mask'].size()}")
        # print(f"wrapper data:\n{data.size()}")
        if self.use_feat_extractor:
            data = self.feature_extractor(data[0].to('cpu'), sampling_rate = 16000, return_tensors = 'pt')
            data = data.input_features.to(self.device)

        # print(f"wrapper feats:\n{data.size()}")

        outputs = self.model.generate(
            input_features = data,
            output_hidden_states = True,
            return_dict_in_generate = True
        )

        decoder_hidden = torch.stack([outputs.decoder_hidden_states[word][self.layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
        decoder_hidden = decoder_hidden.unsqueeze(0)
        # print(outputs.sequences)
        # print(decoder_hidden)
        # print(decoder_hidden.size())
        # quit()
        # print(outputs.sequences)
        # print(f"sequences[0]: {len(outputs.sequences[0])}")
        # print(f"decoder_hidden_states: {len(outputs.decoder_hidden_states)}")
        # # print(f"decoder_hidden_states[0]: {transcript.decoder_hidden_states[0]}")
        # print(f"decoder_hidden_states[0]: {len(outputs.decoder_hidden_states[17])}")
        # print(f"decoder_hidden_states[0][0]: {len(outputs.decoder_hidden_states[17][12])}")
        # print(f"decoder_hidden_states[0][0][0]: {len(outputs.decoder_hidden_states[17][12][0])}")
        # print(f"decoder_hidden_states[0][0][0][0]: {len(outputs.decoder_hidden_states[17][12][0][0])}")

        # print(f"decoder_hidden_states[0][0][1]: {len(transcript.decoder_hidden_states[0][0][1])}")
        # print(f"decoder_hidden_states[0]: {len(transcript.sequences[0])}")
        # print(f"decoder_hidden_states[0][0]: {transcript.sequences[0][0]}")
        # print(f"decoder_hidden_states[0][0]: {len(transcript.sequences[18])}")

        # if self.layer is None:
        #     data = self.model(
        #         input_features = data, 
        #         # attention_mask = data['attention_mask'],
        #         return_dict = True
        #     )
        #     data = data[0]
        # else:
        #     # transcript = self.model.generate(
        #     #     input_features = data, 
        #     #     # decoder_input_ids = torch.tensor([[1, 1]], device = self.device) * self.model.config.decoder_start_token_id,
        #     #     # # attention_mask = data['attention_mask'],
        #     #     # return_dict = True,
        #     #     # output_hidden_states = True
        #     # )
        #     data = self.model(
        #         input_features = data, 
        #         decoder_input_ids = torch.tensor([[1, 1]], device = self.device) * self.model.config.decoder_start_token_id,
        #         # attention_mask = data['attention_mask'],
        #         return_dict = True,
        #         output_hidden_states = True
        #     )
        #     # past_keys = data.past_key_values

        #     # print("\nPast keys:")
        #     # print(f"1st layer: {len(past_keys)}")
        #     # print(f"2nd layer: {len(past_keys[0])}")
        #     # print(past_keys[0][0].size())
        #     # encoder_hidden = data.encoder_hidden_states[self.layer]
        #     decoder_hidden = data.decoder_hidden_states[self.layer]
        #     # print(f"wrapper encoder layer:\n{encoder_hidden.size()}")
        #     # print(f"wrapper decoder layer:\n{decoder_hidden.size()}")
        #     # print(f"wrapper transcript:\n{transcript}")

        #     # print(data)


        #     # for layerID, layer in enumerate(data):
        #     #     print(layerID, layer.size())
        #     # quit()
        #     # print(data)
        #     # print(data["hidden_states"])
        #     # print(f"\nnum hidden states:\n{len(data['hidden_states'])}\n")
        #     # print(f"\nhidden state 0 size:\n{data['hidden_states'][0].size()}\n")
        #     # quit()
        # # data = self.model(data)
        return decoder_hidden



# if __name__ == "__main__":
#     import soundfile as sf
#     import numpy as np
#     import torch
#     import matplotlib.pyplot as plt
#     import sys
#     import speechbrain as sb
#     from speechbrain.processing.features import spectral_magnitude,STFT
#     from scipy.special import expit
#     from torchaudio.transforms import Resample
#     #PATH="results/lstm-hubert-ench-fine-100-2700/save/CKPT+2022-10-30+16-58-24+00/predictor.ckpt"
#     try:
#         from models.will_utils import channel_sort
#     except:
#         from will_utils import channel_sort
#     def compute_feats(wavs):
#         """Feature computation pipeline"""
#         stft = STFT(hop_length=16,win_length=32,sample_rate=16000,n_fft=512,window_fn=torch.hamming_window)
#         feats = stft(wavs)
#         feats = spectral_magnitude(feats, power=0.5)
#         feats = torch.log1p(feats)
#         return feats 


#     def get_img(in_rep,ax,fig):
#         im = ax.imshow(sigmoid_v(in_rep.detach().cpu().numpy()),aspect='auto',norm='linear')
#         fig.colorbar(im,cmap='plasma')


#     def get_img_full(in_rep,ax,fig):
#         im = ax.imshow(in_rep.detach().cpu().numpy(),aspect='auto',norm='linear')
#         fig.colorbar(im,cmap='plasma')

#     def sigmoid(x):
#         return 1 / (1 + expit(-x))
#     sigmoid_v =  np.vectorize(sigmoid)


#     clean_wav = sf.read(sys.argv[1])[0][:,0]
#     noisy_wav= sf.read(sys.argv[2])[0][:,0]
#     #print(clean_wav.shape,noisy_wav.shape)
#     mod= HuBERTWrapper_extractor()
#     mod_full = HuBERTWrapper_full()
#     print(clean_wav.shape,noisy_wav.shape)
#     clean_wav = torch.from_numpy(clean_wav).unsqueeze(0).float()
#     noisy_wav = torch.from_numpy(noisy_wav).unsqueeze(0).float()
#     clean_wav =Resample(44100,16000)(clean_wav)
#     noisy_wav =Resample(44100,16000)(noisy_wav)
#     print(clean_wav.shape,noisy_wav.shape)
#     clean_rep = mod(clean_wav)

#     noisy_rep = mod(noisy_wav)

#     clean_rep_full = mod_full(clean_wav)
#     noisy_rep_full = mod_full(noisy_wav)
    

#     ###################################################################
    
#     fig, ax = plt.subplots(4,2,figsize=(20,20))
#     font = {'family':'times new roman','size'   : 15}

#     plt.rc('font', **font)

#     ax[0,0].plot(clean_wav.squeeze().detach().cpu().numpy())
#     ax[0,0].set_xlabel(r'$n$')
#     ax[0,0].set_ylabel(r'$s[n]$')
#     ax[0,0].set_title(r'$s[n]$')

#     ax[0,1].plot(noisy_wav.squeeze().detach().cpu().numpy())
#     ax[0,1].set_xlabel(r'$n$')
#     ax[0,1].set_ylabel(r'$x[n]$')
#     ax[0,1].set_title(r'$x[n]$')

#     ax[1,0].imshow(compute_feats(clean_wav).T.flipud().squeeze(0),aspect='auto')
#     #plt.xlabel(r'$\mathbf{S}_\mathrm{SG}$')
#     ax[1,0].set_xlabel(r'$T$')
#     ax[1,0].set_ylabel(r'$F_{Hz}$')
#     ax[1,0].set_title(r'$\mathbf{S}_\mathrm{SG}$')

#     ax[1,1].imshow(compute_feats(noisy_wav).T.flipud().squeeze(0),aspect='auto')
#     ax[1,1].set_xlabel(r'$\mathbf{S}_\mathrm{SG}$')
#     ax[1,1].set_xlabel(r'$T$')
#     ax[1,1].set_ylabel(r'$F_{Hz}$')
#     ax[1,1].set_title(r'$\mathbf{X}_\mathrm{SG}$')
    

#     clean_rep_disp = clean_rep.T.squeeze().detach().cpu().numpy()
#     print(clean_rep_disp.shape)
#     chan_idx_clean = channel_sort(clean_rep_disp.T)
#     clean_rep_disp = clean_rep_disp[:,chan_idx_clean]
#     print(clean_rep_disp.shape)
#     ax[2,0].imshow(np.fliplr(sigmoid_v(clean_rep_disp)).T,cmap="plasma",aspect='auto')
#     ax[2,0].set_ylabel(r'$F$')
#     ax[2,0].set_xlabel(r'$T$')
#     ax[2,0].set_title(r'HuBERT $\mathbf{S}_\mathrm{FE}$',fontsize=15)


#     noisy_rep_disp = noisy_rep.T.squeeze().detach().cpu().numpy()
    
#     noisy_rep_disp = noisy_rep_disp[:,chan_idx_clean]
#     ax[2,1].imshow(np.fliplr(sigmoid_v(noisy_rep_disp)).T,cmap="plasma",aspect='auto')
#     ax[2,1].set_ylabel(r'$F$')
#     ax[2,1].set_xlabel(r'$T$')
#     ax[2,1].set_title(r'HuBERT $\mathbf{X}_\mathrm{FE}$')




#     clean_rep_disp = clean_rep_full.T.squeeze().detach().cpu().numpy()
#     print(clean_rep_disp.shape)
#     chan_idx_clean = channel_sort(clean_rep_disp.T)
#     clean_rep_disp = clean_rep_disp[:,chan_idx_clean]
#     print(clean_rep_disp.shape)

#     ax[3,0].imshow(np.fliplr(sigmoid_v(clean_rep_disp)),cmap="plasma",aspect='auto')
#     ax[3,0].set_ylabel(r'$F$')
#     ax[3,0].set_xlabel(r'$T$')
#     ax[3,0].set_title(r'HuBERT $\mathbf{S}_\mathrm{OL}$',fontsize=15)

#     noisy_rep_disp = noisy_rep_full.T.squeeze().detach().cpu().numpy()
#     noisy_rep_disp = noisy_rep_disp[:,chan_idx_clean]
#     print(clean_rep_disp.shape)
#     ax[3,1].imshow(np.fliplr(sigmoid_v(noisy_rep_disp)),cmap="plasma",aspect='auto')
#     ax[3,1].set_ylabel(r'$F$')
#     ax[3,1].set_xlabel(r'$T$')
#     ax[3,1].set_title(r'HuBERT $\mathbf{X}_\mathrm{OL}$')
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig("hubert.png")
#     plt.close()
