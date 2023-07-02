import torch
import torch.nn.functional as F
from torch import Tensor, nn
# try: #look in two places for the HuBERT wrapper
#     from models.huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
#     from models.wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
#     from models.llama_wrapper import LlamaWrapper
# except:
#     from huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
#     from wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
#     from llama_wrapper import LlamaWrapper
# from speechbrain.processing.features import spectral_magnitude,STFT

from models.ni_predictors import PoolAttFF


class Minerva(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, input_dim, rep_dim = None):
        super().__init__()

        if rep_dim is None:
            rep_dim = input_dim

        self.Wx = nn.Linear(input_dim, rep_dim)
        self.Wd = nn.Linear(input_dim, rep_dim)

        self.Wr = nn.Linear(1,1)
        
        self.sm = nn.Softmax(dim = -1)
        
    def forward(self, X: Tensor, D: Tensor, R: Tensor):
        
        # X has dim (batch_size, input_dim)
        # Xw has dim (batch_size, rep_dim)
        Xw = self.Wx(X)
        # D has dim (num_ex, input_dim)
        # Dw has dim (num_ex, rep_dim)
        Dw = self.Wd(D)
        
        # a has dim (batch_size, num_ex)
        a = torch.matmul(Xw, torch.transpose(Dw, dim0 = -2, dim1 = -1))

        # print(f"Xw size: {Xw.size()}")
        # print(f"Dw size: {Dw.size()}")
        # print(f"a size: {a.size()}")

        # R has dim (num_ex, 1)
        # R = r.unsqueeze(1)

        # print(f"R size: {R.size()}")
        # predicts has dim (batch_size, 1)
        predicts = torch.matmul(a, R)
        
        return self.Wr(predicts)



class ExLSTM(nn.Module):
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
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim=512, use_lstm = True
    ):
        super().__init__()

        self.use_lstm = use_lstm

        self.activation = activation(negative_slope=0.3)

        if use_lstm:
            self.blstm = nn.LSTM(
                input_size=dim_extractor,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.1,
                bidirectional=True,
                batch_first=True,
            )
        
        self.attenPool = PoolAttFF(att_pool_dim, output_dim = att_pool_dim)
        self.minerva = Minerva(att_pool_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, D, r,  packed_sequence = True):
        # print(f"LSTM X size: {X.data.size()}")
        # print(f"LSTM D size: {D.data.size()}")
        if self.use_lstm:
            X, _ = self.blstm(X)
            D, _ = self.blstm(D)
        if packed_sequence:
            X, X_len = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
            D, D_len = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)
        # print(f"LSTM unpacked X size: {X.size()}")
        # print(f"LSTM unpacked D size: {D.size()}")
        X = self.attenPool(X)
        D = self.attenPool(D)
        # print(f"LSTM atten unpacked X size: {X.size()}")
        # print(f"LSTM atten unpacked D size: {D.size()}")

        echo = self.minerva(X, D, r)
        # print(f"LSTM echo size: {echo.size()}")
        # out = self.sigmoid(out)

        return echo, None


class MetricPredictorLSTM(nn.Module):
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

    def forward(self, x, packed_sequence = True):
        # print(f"MetricPredictorLSTM: {x.data.size()}")
        out,_ = self.blstm(x)
        # out = out.data
        if packed_sequence:
            # print(out.data.size())
            # print(out.batch_sizes.size())
            # print(out.batch_sizes)
            out, out_len = nn.utils.rnn.pad_packed_sequence(out, batch_first = True)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,_


class MetricPredictorLSTMCombo(nn.Module):
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


class MetricPredictorAttenPool(nn.Module):
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
        self, att_pool_dim=512
    ):
        super().__init__()
        
        self.attenPool = PoolAttFF(att_pool_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        out, out_len = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out, None