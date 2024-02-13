import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.ni_predictors import PoolAttFF
import math


# Activations
class power_activation(nn.Module):

    def __init__(self, p):
        super().__init__()

        self.p = p

    def forward(self, x):
        
        return torch.pow(x, self.p)

class inf_norm_activation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        
        return nn.functional.normalize(x, p = torch.inf, dim = -1)



# Models
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

    def forward(self, X, left = None):
        
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



class ffnn_init(nn.Module):
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
        self.args = args

        if args.restart_model is not None:
            self.load(args.restart_model)

        else:

            if args.which_ear == 'both':
                self.layer0l = nn.Linear(args.feat_dim, args.feat_embed_dim)
                self.layer0r = nn.Linear(args.feat_dim, args.feat_embed_dim)
                # print(f"self.layer0l.weight.size: {self.layer0l.weight.size()}")
                # print(f"self.layer0r.weight.size: {self.layer0r.weight.size()}")
            else:
                self.layer0 = nn.Linear(args.feat_dim, args.feat_embed_dim)
                # print(f"self.layer0.weight.size: {self.layer0.weight.size()}")

            self.do0 = nn.Dropout(p = args.dropout)
            self.activation0 = self.select_activation(args.act0)
            self.layer1 = nn.Linear(args.feat_embed_dim, args.class_embed_dim)
            # print(f"self.layer1.weight.size: {self.layer1.weight.size()}")
            if args.class_embed_dim > 1:
                self.do1 = nn.Dropout(p = args.dropout)
            else:
                self.do1 = nn.Dropout(p = 0)
            self.activation1 = self.select_activation(args.act1)
            self.layer2 = nn.Linear(args.class_embed_dim, 1)
            # print(f"self.layer2.weight.size: {self.layer2.weight.size()}")

            if args.use_layer_norm:
                self.ln0 = nn.LayerNorm(args.input_dim)
                self.ln1 = nn.LayerNorm(args.feat_embed_dim)
                self.ln2 = nn.LayerNorm(1)

            self.sigmoid = nn.Sigmoid()
            

                
    def select_activation(self, acivation_name):
        if acivation_name == 'power':
            return power_activation(self.args.p_factor)
        elif acivation_name == 'sigmoid':
            return nn.Sigmoid()
        elif acivation_name == 'softmax':
            return nn.Softmax(dim = -1)
        elif acivation_name == 'inf_norm':
            return inf_norm_activation()
        elif acivation_name == 'ReLU':
            return nn.ReLU()
        else:
            print(f"{acivation_name} is not known.")

    def initialise_layers(self, initialisation, inits = None):

        # correct_transform = torch.zeros(self.args.class_embed_dim, 1, dtype = torch.float)
        # nn.init.kaiming_uniform_(correct_transform, a=math.sqrt(5))
        # inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
        # X = torch.rand(6, 1)
        # print(X)
        # X = X @ correct_transform.t()
        # print(X)
        # X = X @ inv_correct_transform
        # print(X)

        ex_feats_l, ex_feats_r, ex_correct = inits


        if initialisation == 'minerva':
            ex_feats_l, ex_feats_r, ex_correct = inits
            ex_feats_l = ex_feats_l - ex_feats_l.mean()
            ex_feats_r = ex_feats_r - ex_feats_l.mean()
            # print(f"ex_correct:\n{ex_correct}\n")
            if self.args.which_ear == 'both':
                self.layer0l.weight = nn.Parameter(nn.functional.normalize(self.args.alpha * ex_feats_l, dim = -1))
                self.layer0r.weight = nn.Parameter(nn.functional.normalize(self.args.alpha * ex_feats_r, dim = -1))
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                
            else:
                if self.args.which_ear == 'left':
                    ex_feats = ex_feats_l
                elif self.args.which_ear == 'right':
                    ex_feats = ex_feats_r
                self.layer0.weight = nn.Parameter(nn.functional.normalize(self.args.alpha * ex_feats, dim = -1))
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))

            correct_transform = torch.ones(1, self.args.class_embed_dim, dtype = torch.float)
            if self.args.class_embed_dim > 1:
                nn.init.kaiming_uniform_(correct_transform, a=math.sqrt(5))
            # print(f"correct_transform:\n{correct_transform}\n")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            ex_correct = (ex_correct * 2 - 1) @ correct_transform
            # print(f"ex_correct transformed:\n{ex_correct}\n")

            self.layer1.weight = nn.Parameter(self.args.alpha * ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(self.args.alpha * inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))
            
            # test_ex_correct = self.layer2(ex_correct)
            # print(f"test_ex_correct: \n{test_ex_correct}")
            # quit()


        if initialisation == 'minerva_scaled':
            ex_feats_l_base, ex_feats_r_base, ex_correct_base = inits
            ex_feats_l_base = ex_feats_l_base - ex_feats_l_base.mean()
            ex_feats_r_base = ex_feats_r_base - ex_feats_r_base.mean()

            # print(f"ex_correct:\n{ex_correct}\n")
            if self.args.which_ear == 'both':
                ex_feats_l = nn.functional.normalize(ex_feats_l_base, dim = -1)
                ex_feats_r = nn.functional.normalize(ex_feats_r_base, dim = -1)
                scale0_l = ex_feats_l_base @ ex_feats_l.t()
                scale0_r = ex_feats_r_base @ ex_feats_r.t()
                scale0_l = (scale0_l.sum() - torch.trace(scale0_l)) / (self.args.feat_embed_dim - 1)**2
                scale0_r = (scale0_r.sum() - torch.trace(scale0_r)) / (self.args.feat_embed_dim - 1)**2
                self.layer0l.weight = nn.Parameter(self.args.alpha * ex_feats_l / scale0_l)
                self.layer0r.weight = nn.Parameter(self.args.alpha * ex_feats_r / scale0_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))

                X_l = self.layer0l(ex_feats_l_base)
                X_r = self.layer0r(ex_feats_l_base)
                X = (X_l + X_r) / 2
 
            else:
                if self.args.which_ear == 'left':
                    ex_feats_base = ex_feats_l_base
                elif self.args.which_ear == 'right':
                    ex_feats_base = ex_feats_r_base
                
                ex_feats = nn.functional.normalize(ex_feats_base, dim = -1)
                scale0 = ex_feats_base @ ex_feats.t()
                scale0 = (scale0.sum() - torch.trace(scale0)) / (self.args.feat_embed_dim - 1)**2
                self.layer0.weight = nn.Parameter(self.args.alpha * ex_feats / scale0)
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
            

            correct_transform = torch.ones(1, self.args.class_embed_dim, dtype = torch.float)
            if self.args.class_embed_dim > 1:
                nn.init.kaiming_uniform_(correct_transform, a=math.sqrt(5))
            # print(f"correct_transform:\n{correct_transform}\n")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            ex_correct = (ex_correct_base * 2 - 1) @ correct_transform

            scale1 = (X @ ex_correct).mean()

            # output_l = ex_feats_l_base @ ex_correct
            # output_l = ex_feats_l_base @ ex_correct
            # print(f"ex_correct transformed:\n{ex_correct}\n")

            self.layer1.weight = nn.Parameter(self.args.alpha * ex_correct.t() / scale1)
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            X = self.layer1(X)

            scale2 = (X @ inv_correct_transform.t()).mean()

            self.layer2.weight = nn.Parameter(self.args.alpha * inv_correct_transform / scale2)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))


        if initialisation == 'minerva_scaled2':
            ex_feats_l, ex_feats_r, ex_correct = inits
            ex_feats_l = ex_feats_l - ex_feats_l.mean()
            ex_feats_r = ex_feats_r - ex_feats_l.mean()

            if self.args.which_ear == 'both':
                ex_feats_l = nn.functional.normalize(ex_feats_l, dim = -1)
                ex_feats_r - nn.functional.normalize(ex_feats_r, dim = -1)
                ex_feats_var = (torch.var(ex_feats_l) + torch.var(ex_feats_r)) / 2
                self.layer0l.weight = nn.Parameter(self.args.p_factor * ex_feats_l)
                self.layer0r.weight = nn.Parameter(self.args.p_factor * ex_feats_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
            else:
                if self.args.which_ear == 'left':
                    ex_feats = nn.functional.normalize(ex_feats_l, dim = -1)
                elif self.args.which_ear == 'right':
                    ex_feats - nn.functional.normalize(ex_feats_r, dim = -1)
                elif self.args.which_ear == 'mean':
                    ex_feats = nn.functional.normalize(ex_feats_l + ex_feats_r, dim = -1)
                ex_feats_var = torch.var(ex_feats)
                self.layer0.weight = nn.Parameter(ex_feats)
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                        
            # correct_transform = torch.ones(1, self.args.class_embed_dim, dtype = torch.float)
            # if self.args.class_embed_dim > 1:
            #     nn.init.kaiming_uniform_(correct_transform, a=math.sqrt(5))
            if self.args.class_embed_dim == 1:
                correct_transform = torch.ones(1, 1, dtype = torch.float)
            else:
                correct_transform = torch.rand(1, self.args.class_embed_dim, dtype = torch.float)
                correct_transform = correct_transform - 0.5

            print(f"correct_transform:\n{correct_transform}")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            print(f"inv_correct_transform:\n{inv_correct_transform}")
            ex_correct = (ex_correct * 2 - 1) @ correct_transform

            # Do variance correction
            ex_correct_var = torch.var(ex_correct)
            inv_trans_var = torch.var(inv_correct_transform)
            if self.args.class_embed_dim == 1:
                inv_trans_var = 1
            ex_correct = self.args.p_factor * ex_correct * (ex_feats_var / ex_correct_var)**0.5
            inv_correct_transform = self.args.p_factor * inv_correct_transform * (inv_trans_var / ex_correct_var)**0.5

            self.layer1.weight = nn.Parameter(ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))

            
        if initialisation == 'minerva_scaled3':
            print(f"Doing minerva_scaled3 inititalisation...")
            ex_feats_l_base, ex_feats_r_base, ex_correct_base = inits
            # ex_feats_l_base = ex_feats_l_base - ex_feats_l_base.mean()
            # ex_feats_r_base = ex_feats_r_base - ex_feats_r_base.mean()

            # print(f"ex_correct:\n{ex_correct}\n")
            if self.args.which_ear == 'both':
                ex_feats_l = nn.functional.normalize(ex_feats_l_base, dim = -1)
                ex_feats_r = nn.functional.normalize(ex_feats_r_base, dim = -1)
                out0_l = ex_feats_l_base @ ex_feats_l.t()
                out0_r = ex_feats_r_base @ ex_feats_r.t()
                bias0_l = -out0_l.mean()
                bias0_r = -out0_r.mean()
                scale0_l = out0_l.var()**0.5
                scale0_r = out0_r.var()**0.5
                self.layer0l.weight = nn.Parameter(self.args.alpha * ex_feats_l / scale0_l)
                self.layer0r.weight = nn.Parameter(self.args.alpha * ex_feats_r / scale0_r)
                # self.layer0l.bias = nn.Parameter(bias0_l)
                # self.layer0r.bias = nn.Parameter(bias0_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim, dtype = torch.float))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim, dtype = torch.float))

                X_l = self.layer0l(ex_feats_l_base)
                X_r = self.layer0r(ex_feats_l_base)
                X = (X_l + X_r) / 2
 
            else:
                if self.args.which_ear == 'left':
                    ex_feats_base = ex_feats_l_base
                elif self.args.which_ear == 'right':
                    ex_feats_base = ex_feats_r_base
                
                ex_feats = nn.functional.normalize(ex_feats_base, dim = -1)
                out0 = ex_feats_base @ ex_feats.t()
                bias0 = -out0.mean()
                scale0 = out0.var()**0.5
                self.layer0.weight = nn.Parameter(self.args.alpha * ex_feats / scale0)
                # self.layer0.bias = nn.Parameter(bias0.repeat(self.args.class_embed_dim))
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim, dtype = torch.float))
            

            correct_transform = torch.ones(1, self.args.class_embed_dim, dtype = torch.float)
            if self.args.class_embed_dim > 1:
                nn.init.kaiming_uniform_(correct_transform, a=math.sqrt(5))
            # print(f"correct_transform:\n{correct_transform}\n")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            ex_correct = (ex_correct_base * 2 - 1) @ correct_transform

            out1 = X @ ex_correct
            bias1 = -out1.mean()
            scale1 = out1.var()**0.5

            # output_l = ex_feats_l_base @ ex_correct
            # output_l = ex_feats_l_base @ ex_correct
            # print(f"ex_correct transformed:\n{ex_correct}\n")

            self.layer1.weight = nn.Parameter(self.args.alpha * ex_correct.t() / scale1)
            # self.layer1.bias = nn.Parameter(bias1.repeat(self.args.class_embed_dim))
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            X = self.layer1(X)

            out2 = X @ inv_correct_transform.t()
            bias2 = -out2.mean()
            scale2 = out2.var()**0.5


            self.layer2.weight = nn.Parameter(self.args.alpha * inv_correct_transform / scale2)
            self.layer2.bias = nn.Parameter(bias2)


            

    def forward(self, X, left = False):
        # print(f"X.mean(0): {X.mean(dim = 0)}")
        # print(f"X.mean(1): {X.mean(dim = 1)}")
        # print(f"X.mean: {X.mean()}")

        
        # print(f"X:\n{X}")
        if self.args.normalize:
            X = nn.functional.normalize(X, dim = -1)
        if self.args.use_layer_norm:
            X = self.ln0(X)
        if self.args.which_ear != 'both':
            X = self.layer0(X)
            # print(f"layer0:\n{self.layer0.weight}")
            # print(f"layer0.mean: {self.layer0.weight.mean()}")
        elif left:
            # print(f"layer0l:\n{self.layer0l.weight}")
            # print(f"layer0l.mean: {self.layer0l.weight.mean()}")
            X = self.layer0l(X)
        else:
            # print(f"layer0r:\n{self.layer0r.weight}")
            # print(f"layer0r.mean: {self.layer0r.weight.mean()}")
            X = self.layer0r(X)
        # print(f"X before L0 activation:\n{X}")
        X = self.do0(X)
        X = self.activation0(X)
        # print(f"X after L0 activation:\n{X}")
        if self.args.use_layer_norm:
            X = self.ln1(X)
        # print(f"layer1:\n{self.layer1.weight}")
        # print(f"layer1.mean: {self.layer1.weight.mean()}")
        X = self.layer1(X)
        # print(f"X before L1 activation:\n{X}")
        X = self.do1(X)
        X = self.activation1(X)
        # print(f"X after L1 activation:\n{X}")
        if self.args.use_layer_norm:
            X = self.ln2(X)
        # print(f"layer2:\n{self.layer2.weight}")
        # print(f"layer2.mean: {self.layer2.weight.mean()}")
        X = self.layer2(X)
        # print(X)
        # print(self.sigmoid(X))
        # print(f"X before L2 activation:\n{X}")
        # quit()
            
        return self.sigmoid(X)
    

    def save(self, save_file):

        self.to('cpu')
        torch.save(
            {
                "args": self.args,
                "state_dict": self.state_dict()
            }, 
            save_file
        )
        self.to(self.args.device)

    
    def load(self, load_file):

        self.to('cpu')
        save_dict = torch.load(load_file)
        self.args = save_dict["args"]
        self.__init__(self.args)
        self.load_state_dict(save_dict["state_dict"])
        self.to(self.args.device)






