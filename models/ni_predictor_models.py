import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.ni_predictors import PoolAttFF




class Minerva_with_encoding(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, input_dim, p_factor = 1, rep_dim = None, R_dim = None, use_sm = False):
        super().__init__()
            
        rep_dim = input_dim if rep_dim is None else rep_dim
        self.R_dim = R_dim if R_dim is not None else 4
        self.use_sm = use_sm

        self.Wx = nn.Linear(input_dim, rep_dim)
        self.Wd = nn.Linear(input_dim, rep_dim)
        self.p_factor = p_factor

        # self.Wr = nn.Linear(1, self.R_dim)
        self.We = nn.Linear(self.R_dim, 1)
        
        self.sm = nn.Softmax(dim = -1)

        self.encoding_ids = torch.arange(20).repeat(1, 20)
        encoding_ids = []
        for i in range(1, 21):
            for j in range(i + 1):
                encoding_ids.append(j / i)

        encoding_ids = torch.tensor(encoding_ids, dtype = torch.float).unique()
        encoding_ids, _ = torch.sort(encoding_ids)
        self.encoding_ids = nn.Parameter(encoding_ids, requires_grad = False)
        pos_encoding = self.getPositionEncoding(len(encoding_ids), self.R_dim)
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad = False)
        # print(self.encoding_ids)
        # print(self.encoding_ids.size())
        # print(self.pos_encoding)
        # print(self.pos_encoding.size())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
    def forward(self, X: Tensor, D: Tensor, R: Tensor):

        # X has dim (batch_size, input_dim)
        # D has dim (*, num_ex, input_dim)

        # print(f"X size: {X.size()}")
        # print(f"D size: {D.size()}")


        # print(f"R: {R}")
        encoding_ids = self.encoding_ids.repeat(R.size(0), 1)#.to(self.device)
        # print(f"encoding_ids.size: {encoding_ids.size()}")
        R = R.repeat(1, encoding_ids.size(1))
        # print(f"R.size: {R.size()}")
        # print(torch.argmin(torch.abs(R - encoding_ids), dim = 1))
        pos_ids = torch.argmin(torch.abs(R - encoding_ids), dim = 1)
        # print(f"pos_ids: {pos_ids}")
        R_encoding = self.pos_encoding[pos_ids]
        # print(f"R_encoding: {R_encoding}")
        # R = encoding_ids[0][torch.argmin(torch.abs(R - encoding_ids), dim = 1)]
        # print(f"R: {R}")
        # print(f"encoding_ids: {encoding_ids}")
        # r_range = torch.arange(len(self.encoding_ids), dtype = torch.long, device = self.device)
        # pos_ids = a_range()

        # Xw has dim (batch_size, rep_dim)
        # Dw has dim (*, num_ex, rep_dim)        
        Xw = self.Wx(X)
        Dw = self.Wd(D)
        # print(f"Xw size: {Xw.size()}")
        # print(f"Dw size: {Dw.size()}")
        
        # a has dim (*, batch_size, num_ex)
        a = torch.matmul(Xw, torch.transpose(Dw, dim0 = -2, dim1 = -1))
        a = self.activation(a)
        # print(f"a size: {a.size()}")
        if self.use_sm:
            a = self.sm(a)

        # R has dim (*, num_ex, 1)
        # print(f"R size: {R.size()}")
        # R has dim (*, num_ex, 1)
        echo = torch.matmul(a, R_encoding)
        # print(f"echo size: {echo.size()}")
        
        return self.We(echo)
    
    
    def getPositionEncoding(self, seq_len, d, n = 10000):
        P = torch.zeros((seq_len, d), dtype = torch.float)
        # print(f"seq_len: {seq_len}")
        # print(f"d: {d}")
        for k in range(seq_len):
            for i in torch.arange(int(d / 2)):
                # print(f"i: {i}")
                denominator = torch.pow(n, 2 * i / d)
                P[k, 2 * i] = torch.sin(k / denominator)
                P[k, 2 * i + 1] = torch.cos(k / denominator)
                # print(f"k: {k}, i: {i}, P[k, 2 * i]: {P[k, 2 * i]}, P[k, 2 * i + 1]: {P[k, 2 * i + 1]}")
        return P

    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))



class Minerva(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, input_dim, p_factor = 1, rep_dim = None, R_dim = None, use_sm = False):
        super().__init__()
            
        rep_dim = input_dim if rep_dim is None else rep_dim
        R_dim = R_dim if R_dim is not None else 1
        self.use_sm = use_sm

        self.Wx = nn.Linear(input_dim, rep_dim)
        self.Wd = nn.Linear(input_dim, rep_dim)
        self.p_factor = p_factor

        self.Wr = nn.Linear(1, R_dim)
        self.We = nn.Linear(R_dim, 1)
        
        self.sm = nn.Softmax(dim = -1)
        
    def forward(self, X: Tensor, D: Tensor, R: Tensor):

        # X has dim (batch_size, input_dim)
        # D has dim (*, num_ex, input_dim)

        # print(f"X size: {X.size()}")
        # print(f"D size: {D.size()}")
        
        # Xw has dim (batch_size, rep_dim)
        # Dw has dim (*, num_ex, rep_dim)        
        Xw = self.Wx(X)
        Dw = self.Wd(D)
        Rw = self.Wr(R)
        # print(f"Xw size: {Xw.size()}")
        # print(f"Dw size: {Dw.size()}")
        
        # a has dim (*, batch_size, num_ex)
        # print(f"Xw.size: {Xw.size()}")
        # print(f"Dw.size: {Dw.size()}")
        a = torch.matmul(Xw, torch.transpose(Dw, dim0 = -2, dim1 = -1))
        a = self.activation(a)
        # print(f"a size: {a.size()}")
        if self.use_sm:
            a = self.sm(a)

        # R has dim (*, num_ex, 1)
        # print(f"R size: {R.size()}")
        # R has dim (*, num_ex, 1)
        echo = torch.matmul(a, Rw)
        # print(f"echo size: {echo.size()}")
        
        return self.We(echo)
    
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
        # D has dim (*, num_ex, input_dim)
        
        # Xw has dim (batch_size, rep_dim)
        # Dw has dim (num_minervas, num_ex, rep_dim)        
        Xw = self.Wx(X)
        Dw = self.Wd(D)

        # R has dim (num_minervas, num_ex, 1)
        
        # a has dim (num_minervas, num_ex, batch_size)
        a = torch.matmul(Dw, torch.transpose(Xw, dim0 = -2, dim1 = -1))
        # a has dim (num_minervas, batch_size, num_ex)
        a = torch.transpose(a, dim0 = -2, dim1 = -1)
        a = self.activation(a)
        
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
        minerva_dim = None,
        minerva_R_dim = None,
        use_r_encoding = False
    ):
        super().__init__()

        self.use_lstm = use_lstm
        # self.activation = activation(negative_slope=0.3)

        minerva_dim = dim_extractor if minerva_dim is None else minerva_dim
        minerva_R_dim = minerva_R_dim if minerva_R_dim is not None else 1

        self.layer_weights = nn.Parameter(torch.ones(num_layers, dtype = torch.float))
        self.att_pool_dim = att_pool_dim
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

        if use_r_encoding:
            self.minerva = Minerva_with_encoding(
            att_pool_dim, 
            p_factor = p_factor, 
            rep_dim = minerva_dim, 
            R_dim = minerva_R_dim
        )
        else:
            self.minerva = Minerva(
                att_pool_dim, 
                p_factor = p_factor, 
                rep_dim = minerva_dim, 
                R_dim = minerva_R_dim
            )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, D, r, packed_sequence = True, num_minervas = 1):

        # X has dim (batch size, time (padded), input_dim, layers)
        X, X_len = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
        # D has dim (ex size, time (padded), input_dim, layers)
        D, D_len = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)

        # X has new dim (batch size, time (padded), input_dim)
        X = X @ self.sm(self.layer_weights)
        # D has new dim (ex size, time (padded), input_dim)
        D = D @ self.sm(self.layer_weights)

        X = nn.utils.rnn.pack_padded_sequence(X, lengths = X_len, batch_first = True, enforce_sorted = False)
        D = nn.utils.rnn.pack_padded_sequence(D, lengths = D_len, batch_first = True, enforce_sorted = False)

        if self.use_lstm:
            X, _ = self.blstm(X)
            D, _ = self.blstm(D)
        if packed_sequence:
            X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
            D, _ = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)
        
        # X has new dim (batch_size, att_pool_dim)
        X = self.attenPool(X)
        # X has new dim (ex_size, att_pool_dim)
        D = self.attenPool(D)

        if num_minervas > 1:
            # X has new dim (num_minervas, ex_size, att_pool_dim)
            D = D.view(num_minervas, -1, self.att_pool_dim)
            # X has new dim (num_minervas, ex_size, 1)
            r = r.view(num_minervas, -1, 1)

        # full_echos has dim (*, batch_size, 1)
        full_echo = self.minerva(X, D, r)
        if num_minervas > 1:
            echo = full_echo.mean(dim = 0)
        else:
            echo = full_echo
        # print(f"echo size: {echo.size()}")

        return self.sigmoid(echo), full_echo
    



class ExLSTM_log(nn.Module):
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
        minerva_dim = None,
        log_corr = 0.1
    ):
        super().__init__()

        self.use_lstm = use_lstm
        # self.activation = activation(negative_slope=0.3)

        minerva_dim = dim_extractor if minerva_dim is None else minerva_dim
        self.log_corr = log_corr

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

    def forward(self, X, D, r, packed_sequence = True, num_minervas = None):

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

        # r has dim (ex_size)
        # inverse sigmoid
        r = torch.clamp(r, min = self.log_corr, max = 1 - self.log_corr)
        r = torch.log(r / (1 - r))

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


class ExLSTM_div(nn.Module):
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
        minerva_dim = None,
        log_corr = 0.1
    ):
        super().__init__()

        self.use_lstm = use_lstm
        # self.activation = activation(negative_slope=0.3)

        minerva_dim = dim_extractor if minerva_dim is None else minerva_dim
        self.log_corr = log_corr

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
        self.minerva = Minerva(att_pool_dim, p_factor = p_factor, rep_dim = minerva_dim, use_sm = True)
        
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

        # r has dim (ex_size)
        r = r * 2 - 1

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

    def forward(self, X, D, r,  packed_sequence = True, num_minervas = None):

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
    

class ffnn(nn.Module):

    def __init__(
            self,
            input_dim = 768, 
            embed_dim = 768,
            output_dim = 768,
            dropout = 0.0,
            activation = nn.ReLU()
            ):
        super().__init__()

        self.fnn_stack = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Dropout(p = dropout),
            activation,
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p = dropout),
            activation,
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, features):
        
        return self.fnn_stack(features)

    

class wordLSTM(nn.Module):
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
        
        self.layer_weights = nn.Parameter(torch.ones(num_layers, dtype = torch.float))

        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        self.f = nn.Linear(att_pool_dim, 1)
        self.sm = nn.Softmax(dim = 0)
        self.sigmoid = nn.Sigmoid()

        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, X, D, r, packed_sequence = True):

        # X has dim (batch size, num_words (padded), input_dim, layers)
        X, X_len = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
        print(f"X_len: {X_len}")

        # # D has dim (ex size, num_words (padded), input_dim, layers)
        # D, D_len = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)

        # X has new dim (batch size, num_words (padded), input_dim)
        X = X @ self.sm(self.layer_weights)
        print(f"X size: {X.size()}")
        X = nn.utils.rnn.pack_padded_sequence(X, lengths = X_len, batch_first = True, enforce_sorted = False)
        X, _ = self.blstm(X)
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first = True)
        # X has new dim (batch size, num_words (padded), 1)
        print(f"X size post lstm: {X.size()}")
        X = self.f(X)
        X = self.sigmoid(X)
        print(f"X size post class: {X.size()}")
        # X_mask = torch.zeros_like(X, dtype = torch.bool)
        X_mask = (torch.arange(X_len.max())[None, :] < X_len[:, None]).unsqueeze(-1).to(self.device)
        print(f"X_mask size: {X_mask.size()}")
        print(f"X_mask: {X_mask}")
        print(f"X_len: {X_len}")
        # correct_words has dim batch_size
        correct_words = (X * X_mask).sum(dim = -2)
        print(f"correct_words: {correct_words}")
        print(f"correct_words size: {correct_words.size()}")
        # prop_correct has dim ()
        prop_correct = correct_words / X_len.unsqueeze(-1).to(self.device)
        print(f"prop_correct: {prop_correct}")
        print(f"prop_correct size: {prop_correct.size()}")

        # # D has new dim (ex size, num_words (padded), input_dim)
        # D = D @ self.sm(self.layer_weights)
        # print(f"X size: {X.size()}")
        # X = X.squeeze(-1)


        # D = nn.utils.rnn.pack_padded_sequence(D, lengths = D_len, batch_first = True, enforce_sorted = False)

        # if self.use_lstm:
        #     D, _ = self.blstm(D)
        # if packed_sequence:
        #     D, _ = nn.utils.rnn.pad_packed_sequence(D, batch_first = True)
        # X = self.attenPool(X)
        # D = self.attenPool(D)

        # echo = self.minerva(X, D, r)
        # print(echo)

        # echo = torch.clamp(echo, min = 0, max = 1)

        # print(echo)
        # print(self.sigmoid(echo))

        return prop_correct, None