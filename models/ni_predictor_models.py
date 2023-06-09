import torch
import torch.nn.functional as F
from torch import Tensor, nn
try: #look in two places for the HuBERT wrapper
    from models.huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
    from models.wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
    from models.llama_wrapper import LlamaWrapper
except:
    from huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
    from wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
    from llama_wrapper import LlamaWrapper
from speechbrain.processing.features import spectral_magnitude,STFT

from models.ni_predictors import PoolAttFF


class MetricPredictor(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim=512
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        self.attenPool = PoolAttFF(att_pool_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        out,_ = self.blstm(x)
        print(f"out:\n{out}")
        out = out.data
        print(f"out.size: {out.size()}")
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,_


class MetricPredictorCombo(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=1024, hidden_size=1024//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)
        
        self.blstm_last = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size//2,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        self.blstm_encoder  = nn.LSTM(
            input_size=dim_extractor//2,
            hidden_size=hidden_size//2,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats_full, feats_extact):

        out_full,_ = self.blstm_last(feats_full)
        out_extract,_ = self.blstm_encoder(feats_extact)

        #print(out_full.shape,out_extract.shape)
        out = torch.cat((out_full,out_extract),dim=2)
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,_


