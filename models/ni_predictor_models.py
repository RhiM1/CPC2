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
            
        rep_dim = input_dim if rep_dim is None else rep_dim

        self.Wx = nn.Linear(input_dim, rep_dim)
        self.Wd = nn.Linear(input_dim, rep_dim)
        self.p_factor = p_factor

        self.Wr = nn.Linear(1,1)
        
        self.sm = nn.Softmax(dim = -1)
        
    def forward(self, X: Tensor, D: Tensor, R: Tensor):

        # X has dim (batch_size, input_dim)
        # D has dim (num_minervas, num_ex, input_dim)
        
        # Xw has dim (batch_size, rep_dim)
        # Dw has dim (num_minervas, num_ex, rep_dim)        
        Xw = self.Wx(X)
        Dw = self.Wd(D)
        
        # a has dim (batch_size, )
        a = torch.matmul(Xw, torch.transpose(Dw, dim0 = -2, dim1 = -1))
        # a = self.activation(a)
        # a = self.sm(a)
        # a = nn.functional.normalize(a, dim = 1, p = 1)
        echo = torch.matmul(a, R)
        
        return self.Wr(echo)
    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))




class Minerva2(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, input_dim, p_factor = 1, rep_dim = None):
        super().__init__()
            
        rep_dim = input_dim if rep_dim is None else rep_dim

        self.Wx = nn.Linear(input_dim, rep_dim)
        self.Wd = nn.Linear(input_dim, rep_dim)
        self.p_factor = p_factor

        self.Wr = nn.Linear(1,1)
        
        self.sm = nn.Softmax(dim = -1)
        
    def forward(self, X: Tensor, D: Tensor, R: Tensor):

        # X has dim (batch_size, input_dim)
        # D has dim (num_minervas, num_ex, input_dim)
        
        # Xw has dim (batch_size, rep_dim)
        # Dw has dim (num_minervas, num_ex, rep_dim)        
        Xw = self.Wx(X)
        Dw = self.Wd(D)
        
        # a has dim (num_minervas, num_ex, batch_size)
        a = torch.matmul(Dw, torch.transpose(Xw, dim0 = -2, dim1 = -1))
        # a has dim (num_minervas, batch_size, num_ex)
        a = torch.transpose(a, dim0 = -2, dim1 = -1)
        a = self.activation(a)
        
        # R has dim (num_minervas, num_ex, 1)
        # echo has dim (num_minervas, batch_size)
        # print(f"a.size: {a.size()}")
        # print(f"R.size: {R.size()}")
        echo = torch.matmul(a, R)
        
        return self.Wr(echo)
    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))


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

        # self.activation = activation(negative_slope=0.3)

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







class ExLSTM_layers(nn.Module):
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
        self, 
        dim_extractor=512, 
        hidden_size=512//2, 
        # activation=nn.LeakyReLU, 
        att_pool_dim=512, 
        use_lstm = True, 
        num_layers = 12, 
        p_factor = 1,
        minerva_dim = None
    ):
        super().__init__()

        self.use_lstm = use_lstm
        # self.activation = activation(negative_slope=0.3)

        minerva_dim = dim_extractor if minerva_dim is None else minerva_dim

        self.layer_weights = nn.Parameter(torch.ones(num_layers, dtype = torch.float))
        self.sm = nn.Softmax(dim = 0)

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
        self.minerva = Minerva(att_pool_dim, p_factor = p_factor, rep_dim = minerva_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, D, r, packed_sequence = True):

        # X has dim (batch size, time (padded), input_dim, layers)
        X, X_len = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
        # D has dim (ex size, time (padded), input_dim, layers)
        D, D_len = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)

        # X has new dim (batch size, time (padded), input_dim)
        X = X @ self.sm(self.layer_weights)
        # D has new dim (ex size, time (padded), input_dim)
        D = D @ self.sm(self.layer_weights)
        # print(f"X size: {X.size()}")
        # X = X.squeeze(-1)

        X = nn.utils.rnn.pack_padded_sequence(X, lengths = X_len, batch_first = True, enforce_sorted = False)
        D = nn.utils.rnn.pack_padded_sequence(D, lengths = D_len, batch_first = True, enforce_sorted = False)

        if self.use_lstm:
            X, _ = self.blstm(X)
            D, _ = self.blstm(D)
        if packed_sequence:
            X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
            D, _ = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)
        X = self.attenPool(X)
        D = self.attenPool(D)

        echo = self.minerva(X, D, r)

        # echo = torch.clamp(echo, min = 0, max = 1)

        # print(echo)
        # print(self.sigmoid(echo))

        return self.sigmoid(echo), None
    



class ExLSTM_std(nn.Module):
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
        self, 
        dim_extractor=512, 
        hidden_size=512//2, 
        att_pool_dim=512, 
        use_lstm = True, 
        num_layers = 12, 
        p_factor = 1,
        num_minervas = 1,
        minerva_dim = None
    ):
        super().__init__()

        self.use_lstm = use_lstm
        # self.activation = activation(negative_slope=0.3)

        self.att_pool_dim = att_pool_dim
        minerva_dim = dim_extractor if minerva_dim is None else minerva_dim
        self.num_minervas = num_minervas

        self.layer_weights = nn.Parameter(torch.ones(num_layers, dtype = torch.float))
        self.sm = nn.Softmax(dim = 0)

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

        # self.minervas = []
        # for i in range(num_minervas):
        #     self.minervas.append(Minerva(att_pool_dim, p_factor = p_factor, rep_dim = minerva_dim))
        
        self.minerva = Minerva2(att_pool_dim, p_factor = p_factor, rep_dim = minerva_dim)

        self.calibrate = nn.Linear(1, 1)

        self.sigmoid = nn.Sigmoid()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, X, D, r, packed_sequence = True):

        # X has dim (batch size, time (padded), input_dim, layers)
        X, X_len = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
        # D has dim (ex size, time (padded), input_dim, layers)
        D, D_len = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)

        # X has new dim (batch size, time (padded), input_dim)
        X = X @ self.sm(self.layer_weights)
        # D has new dim (ex size, time (padded), input_dim)
        D = D @ self.sm(self.layer_weights)
        # print(f"X size: {X.size()}")
        # X = X.squeeze(-1)

        X = nn.utils.rnn.pack_padded_sequence(X, lengths = X_len, batch_first = True, enforce_sorted = False)
        D = nn.utils.rnn.pack_padded_sequence(D, lengths = D_len, batch_first = True, enforce_sorted = False)

        if self.use_lstm:
            X, _ = self.blstm(X)
            D, _ = self.blstm(D)
        if packed_sequence:
            X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
            D, _ = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)
        X = self.attenPool(X)
        D = self.attenPool(D)

        # print(f"D.size: {D.size()}")
        D = D.view(self.num_minervas, -1, self.att_pool_dim)
        # print(f"D.size: {D.size()}")
        # print(f"r.size: {r.size()}")
        r = r.view(self.num_minervas, -1, 1)
        # print(f"r.size: {r.size()}")


        # echos = torch.zeros(self.num_minervas, X.size(0), dtype = torch.float, device = self.device)


        # for i, minerva in enumerate(self.minervas):
        #     echos[i] = minerva(X, D[i], r[i])
        
        echos = self.minerva(X, D, r)
        # print(f"echos.size: {echos.size()}")
        preds = torch.std(echos, dim = 0)
        # print(f"preds.size: {echos.size()}")


        # echo = torch.clamp(echo, min = 0, max = 1)

        # print(echo)
        # print(self.sigmoid(echo))

        return self.calibrate(preds), None


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
        self, dim_extractor=512, hidden_size=512//2, att_pool_dim=512, use_lstm = True, p_factor = 1, minerva_dim = None
    ):
        super().__init__()

        minerva_dim = dim_extractor if minerva_dim is None else minerva_dim

        self.use_lstm = use_lstm

        # self.activation = activation(negative_slope=0.3)

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
        self.minerva = Minerva(att_pool_dim, p_factor = p_factor, rep_dim = minerva_dim)
        
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