import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.ni_predictors import PoolAttFF




class ffnn(nn.Module):

    def __init__(
            self,
            input_dim = 768, 
            embed_dim = 768,
            output_dim = 1,
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



class Minerva(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, p_factor = 1):
        super().__init__()

        self.p_factor = p_factor
        
    def forward(self, X: Tensor, D: Tensor, r: Tensor, p = None):

        p = p if p is not None else self.p_factor
        # print(f"X:\n{X}\n")
        # print(f"D:\n{X}\n")
        
        a = torch.matmul(
            nn.functional.normalize(X, dim = 1), 
            nn.functional.normalize(D, dim = 1).transpose(dim0 = -2, dim1 = -1)
        )
        # print(f"s:\n{a}\n")
        a = self.activation(a, p)
        # print(f"a:\n{a}\n")
        # print(f"r:\n{r}\n")
        echo = torch.matmul(a, r)
        # print(f"e:\n{echo}\n")
        
        return echo
    
    def activation(self, s, p):

        return torch.mul(torch.pow(torch.abs(s), p), torch.sign(s))



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



class ffnn_wrapper(nn.Module):
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
        self, args
    ):
        super().__init__()

        self.device = args.device

        self.f = ffnn(
            input_dim = args.feat_dim,
            embed_dim = args.hidden_size,
            output_dim = 1,
            activation = args.activation,
            dropout = args.dropout
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        
        # X_out = X.data.to(self.device)
        X = self.f(X)

        return self.sigmoid(X)



class minerva_wrapper(nn.Module):
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
        args,
        ex_feats_l = None,
        ex_feats_r = None,
        ex_correct = None,
    ):
        super().__init__()

        self.train_ex_class = args.train_ex_class
        self.use_g = args.use_g
        self.device = args.device

        if ex_feats_l is not None:
            self.Dl = nn.Parameter(ex_feats_l, requires_grad = False)
            self.Dr = nn.Parameter(ex_feats_r, requires_grad = False)
        else:
            self.Dl = None
            self. Dr = None

        if ex_correct is not None:
            self.r = nn.Parameter(ex_correct, requires_grad = args.train_ex_class)

        if self.use_g:
            feat_embed_dim = args.feat_dim if args.feat_embed_dim is None else args.feat_embed_dim
            self.g = nn.Linear(in_features = args.feat_dim, out_features = feat_embed_dim)

        self.f = Minerva(args.p_factor)
        self.h = nn.Linear(in_features = 1, out_features = 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, D = None, r = None, left = False):

        if left:
            D = D if D is not None else self.Dl
        else:
            D = D if D is not None else self.Dr
        r = r if r is not None else self.r

        # X_out = X.data.to(self.device)

        r = r * 2 - 1

        if self.use_g:
            X = self.g(X)
            D = self.g(D)

        echo = self.f(X, D, r)

        echo = self.h(echo)
        # print(f"output:\n{echo}\n")

        return self.sigmoid(echo)


  