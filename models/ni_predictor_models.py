import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.ni_predictors import PoolAttFF


class Minerva(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, input_dim, p_factor = 1, rep_dim = None):
        super().__init__()

        if rep_dim is None:
            rep_dim = input_dim

        self.Wx = nn.Linear(input_dim, rep_dim)
        self.Wd = nn.Linear(input_dim, rep_dim)
        self.p_factor = p_factor

        self.Wr = nn.Linear(1,1)
        
        self.sm = nn.Softmax(dim = -1)
        
    def forward(self, X: Tensor, D: Tensor, R: Tensor):
        
        Xw = self.Wx(X)
        Dw = self.Wd(D)
        
        a = torch.matmul(Xw, torch.transpose(Dw, dim0 = -2, dim1 = -1))
        a = self.activation(a)
        a = nn.functional.normalize(a, dim = 1, p = 1)
        echo = torch.matmul(a, R) / R.sum()
        
        return echo
    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))



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
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim=512, use_lstm = True, p_factor = 1
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
        self.minerva = Minerva(att_pool_dim, p_factor = p_factor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, D, r,  packed_sequence = True):

        if self.use_lstm:
            X, _ = self.blstm(X)
            D, _ = self.blstm(D)
        if packed_sequence:
            X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
            D, _ = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)
        X = self.attenPool(X)
        D = self.attenPool(D)

        echo = self.minerva(X, D, r)

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
        out,_ = self.blstm(x)
        if packed_sequence:
            out, out_len = nn.utils.rnn.pad_packed_sequence(out, batch_first = True)
        out = self.attenPool(out)
        out = self.sigmoid(out)

        return out,_
    

    
class MetricPredictorLSTM_layers(nn.Module):
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
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim=512, num_layers = 12
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        # self.layer_weights = torch.nn.Linear(
        #     in_features = num_layers,
        #     out_features = 1,
        #     bias = False
        # )

        self.layer_weights = nn.Parameter(torch.ones(num_layers, dtype = torch.float))
        self.sm = nn.Softmax(dim = 0)

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

    def forward(self, X, packed_sequence = True):

        # X has dim (batch size, time (padded), input_dim, layers)
        X, X_len = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)

        # X has new dim (batch size, time (padded), input_dim)
        X = X @ self.sm(self.layer_weights)
        # print(f"X size: {X.size()}")
        # X = X.squeeze(-1)

        X = nn.utils.rnn.pack_padded_sequence(X, lengths = X_len, batch_first = True, enforce_sorted = False)

        X, _ = self.blstm(X)
        if packed_sequence:
            X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)

        # X has new dim (batch_size, 1)
        X = self.attenPool(X)
        X = self.sigmoid(X)

        return X, None


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

        out = torch.cat((out_full,out_extract),dim=2)
        out = self.attenPool(out)
        out = self.sigmoid(out)

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
        
        out, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        out = self.attenPool(out)
        out = self.sigmoid(out)

        return out, None