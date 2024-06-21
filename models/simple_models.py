import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.ni_predictors import PoolAttFF
import math
from scipy.optimize import curve_fit
import numpy as np


def logit_func(x,a,b):
    return 1/(1+np.exp(-(a*x+b)))


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




class minerva_wrapper2(nn.Module):

    def __init__(
        self, 
        args,
        ex_feats_l = None,
        ex_feats_r = None,
        ex_correct = None,
    ):
        super().__init__()

        self.train_ex_class = args.train_ex_class
        self.device = args.device
        self.feat_embed_dim = args.feat_embed_dim
        self.class_embed_dim = args.class_embed_dim

        if ex_feats_l is not None:
            self.Dl = nn.Parameter(ex_feats_l, requires_grad = False)
            self.Dr = nn.Parameter(ex_feats_r, requires_grad = False)
        else:
            self.Dl = None
            self. Dr = None

        if ex_correct is not None:
            self.r = nn.Parameter(ex_correct, requires_grad = args.train_ex_class)

        if self.feat_embed_dim is not None:
            # feat_embed_dim = args.feat_dim if args.feat_embed_dim is None else args.feat_embed_dim
            self.g = nn.Linear(in_features = args.feat_dim, out_features = self.feat_embed_dim)

        self.f = Minerva(args.p_factor)
        if self.class_embed_dim is not None:
            self.h = nn.Linear(in_features = 1, out_features = self.class_embed_dim)
        # self.h = nn.Linear(in_features = 1, out_features = 1)
        
        self.sigmoid = nn.Sigmoid()

        
    def get_regression(self, feats_l, feats_r, targets):
        # logistic mapping curve fit to get the a and b parameters
        # popt,_ = curve_fit(logit_func, val_predictions, val_gt)

        # print(f"feats: {feats}")
        # print(f"targets: {targets}")
        x = self.forward(feats_l.to(self.device), left = True)
        x_l = x['logits'].to('cpu').squeeze()
        x = self.forward(feats_r.to(self.device), left = False)
        x_r = x['logits'].to('cpu').squeeze()
        x = torch.maximum(x_l, x_r)
        # print(f"logits: {x}")
        # print(f"targets numpy: {targets[:, 0].numpy()}")

        # x_bar = x.mean().item()
        # a0 = 1 / (x - x_bar).abs().max().item()
        # b0 = -x_bar * a0
        a0 = 1/ x.abs().max().item()
        b0 = 0

        # b0 = -x.mean().item()
        # a0 = 1 / (x + b0).abs().max().item()
        # b0 = b0 * a0
        # a0 = 1 / x.abs().max().item()
        # print(f"x numpy: {x.numpy()}")

        # print(f"correctness:\n{x}")
        # print(f"corrected correctness:\n{x * a0 + b0}")
        # print(f"targets[0]:\n{targets}")
        # print(f"a0: {a0}")
        # print(f"b0: {b0}")
        
        popt, _ = curve_fit(logit_func, x.numpy(), targets.squeeze().numpy(), p0 = (a0, b0)) # , 
        a, b = popt
        return a, b
    

    def forward(self, X, D = None, r = None, left = False):

        if left:
            D = D if D is not None else self.Dl
        else:
            D = D if D is not None else self.Dr
        r = r if r is not None else self.r

        # X_out = X.data.to(self.device)

        r = r * 2 - 1

        if self.feat_embed_dim is not None:
            X = self.g(X)
            D = self.g(D)

        echo = self.f(X, D, r)

        if self.class_embed_dim is not None:
            echo = self.h(echo)
        # print(f"output:\n{echo}\n")

        output = {
            'logits': echo,
            'preds': self.sigmoid(echo)
        }

        return output






class minerva_wrapper3(nn.Module):

    def __init__(
        self, 
        args,
        ex_feats_l = None,
        ex_feats_r = None,
        ex_correct = None,
    ):
        super().__init__()

        self.train_ex_class = args.train_ex_class
        self.device = args.device
        self.feat_embed_dim = args.feat_embed_dim
        self.class_embed_dim = args.class_embed_dim

        if ex_feats_l is not None:
            self.Dl = nn.Parameter(ex_feats_l, requires_grad = False)
            self.Dr = nn.Parameter(ex_feats_r, requires_grad = False)
        else:
            self.Dl = None
            self. Dr = None

        if ex_correct is not None:
            self.r = nn.Parameter(ex_correct, requires_grad = args.train_ex_class)

        if self.feat_embed_dim is not None:
            # feat_embed_dim = args.feat_dim if args.feat_embed_dim is None else args.feat_embed_dim
            self.g_q = nn.Linear(in_features = args.feat_dim, out_features = self.feat_embed_dim, bias = False)
            self.g_k = nn.Linear(in_features = args.feat_dim, out_features = self.feat_embed_dim, bias = False)
        
        self.do = nn.Dropout(p = args.dropout)

        self.f = Minerva(args.p_factor)
        if self.class_embed_dim is not None:
            self.h = nn.Linear(in_features = 1, out_features = self.class_embed_dim)
        # self.h = nn.Linear(in_features = 1, out_features = 1)
        
        self.sigmoid = nn.Sigmoid()

        
    def get_regression(self, feats_l, feats_r, targets):
        # logistic mapping curve fit to get the a and b parameters
        # popt,_ = curve_fit(logit_func, val_predictions, val_gt)

        # print(f"feats: {feats}")
        # print(f"targets: {targets}")
        x = self.forward(feats_l.to(self.device), left = True)
        x_l = x['logits'].to('cpu').squeeze()
        x = self.forward(feats_r.to(self.device), left = False)
        x_r = x['logits'].to('cpu').squeeze()
        x = torch.maximum(x_l, x_r)
        # print(f"logits: {x}")
        # print(f"targets numpy: {targets[:, 0].numpy()}")

        # x_bar = x.mean().item()
        # a0 = 1 / (x - x_bar).abs().max().item()
        # b0 = -x_bar * a0
        a0 = 1/ x.abs().max().item()
        b0 = 0

        # b0 = -x.mean().item()
        # a0 = 1 / (x + b0).abs().max().item()
        # b0 = b0 * a0
        # a0 = 1 / x.abs().max().item()
        # print(f"x numpy: {x.numpy()}")

        # print(f"correctness:\n{x}")
        # print(f"corrected correctness:\n{x * a0 + b0}")
        # print(f"targets[0]:\n{targets}")
        # print(f"a0: {a0}")
        # print(f"b0: {b0}")
        
        popt, _ = curve_fit(logit_func, x.numpy(), targets.squeeze().numpy(), p0 = (a0, b0)) # , 
        a, b = popt
        return a, b
    

    def forward(self, X, D = None, r = None, left = False):

        if left:
            D = D if D is not None else self.Dl
        else:
            D = D if D is not None else self.Dr
        r = r if r is not None else self.r

        # X_out = X.data.to(self.device)

        r = r * 2 - 1

        if self.feat_embed_dim is not None:
            X = self.g_q(X)
            D = self.g_k(D)
            
        X = self.do(X)
        D = self.do(D)

        echo = self.f(X, D, r)

        if self.class_embed_dim is not None:
            echo = self.h(echo)
        # print(f"output:\n{echo}\n")

        output = {
            'logits': echo,
            'preds': self.sigmoid(echo)
        }

        return output



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

            
    def get_regression(self, feats_l, feats_r, targets):
        
        print(f"Starting weights regression...")
        
        self.to(self.device)
        # print(f"feats_l:\n{feats_l}")
        # print(f"feats_r:\n{feats_r}")
        
        x_l = self.forward(feats_l.to(self.device), left = True)
        # print(f"x_l:\n{x_l}") # dodgy
        x_r = self.forward(feats_r.to(self.device), left = False)
        # print(f"x_r:\n{x_r}") # ok
        # x_l_r = self.forward(feats_l.to(self.device), left = False)
        # print(f"x_l_r:\n{x_l_r}") # ok
        # x_r_l = self.forward(feats_r.to(self.device), left = True)
        # print(f"x_r_l:\n{x_r_l}") # dodgy
        # x_l_r = self.forward(feats_l.to(self.device), left = False)
        # x_r_l = self.forward(feats_r.to(self.device), left = True)
        # x_l_l = x_l_l['logits']
        # x_r_r = x_r_r['logits']
        # x_l_r = x_l_r['logits']
        # x_r_l = x_r_l['logits']
        # print(f"x_l_l mean: {x_l_l.mean().item()}, x_r_r mean: {x_r_r.mean().item()}")
        # print(f"x_l_r mean: {x_l_r.mean().item()}, x_r_l mean: {x_r_l.mean().item()}")
        # quit()
        x = torch.maximum(x_l['logits'], x_r['logits']).to('cpu').squeeze()
        targets = targets.squeeze()

        x_bar = x.mean().item()
        # print(f"x:\n{x}")
        # print(f"x_bar: {x_bar}")
        # print(f"layer0l.weight:\n{self.layer0l.weight}")
        # print(f"layer0r.weight:\n{self.layer0.weight}")
        if (x - x_bar).abs().max().item() == 0: # usually due to ReLU acting on EVERYTHING
            a0 = 1
        else:
            a0 = 1 / (x - x_bar).abs().max().item()
        b0 = -x_bar * a0
        # quit()

        # print(f"x:\n{x}")
        
        popt, _ = curve_fit(logit_func, x.detach().numpy(), targets.detach().numpy(), p0 = (a0, b0)) # , 
        a, b = popt
        print(f"corrections for intialisations - a: {a}, b: {b}")
        
        print(f"Finished weights regression.")
        return a, b

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
        self.eval()
        # ex_feats_l, ex_feats_r, ex_correct = inits

        
        if initialisation == 'minerva4':
            ex_feats_l_base, ex_feats_r_base, ex_correct_base = inits
            ex_feats_l = ex_feats_l_base.detach()
            ex_feats_r = ex_feats_r_base.detach()
            if self.args.which_ear == 'both':
                ex_feats_l = nn.functional.normalize(ex_feats_l)
                ex_feats_r = nn.functional.normalize(ex_feats_r)
                ex_feats_norm_l = ex_feats_l.norm() / (ex_feats_l.size(0) * ex_feats_l.size(1))
                ex_feats_norm_r = ex_feats_r.norm() / (ex_feats_r.size(0) * ex_feats_l.size(1))
                self.layer0l.weight = nn.Parameter(ex_feats_l / ex_feats_norm_l) 
                self.layer0r.weight = nn.Parameter(ex_feats_r / ex_feats_norm_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
            else:
                if self.args.which_ear == 'left':
                    ex_feats = nn.functional.normalize(ex_feats_l, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'right':
                    ex_feats - nn.functional.normalize(ex_feats_r, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'mean':
                    ex_feats = nn.functional.normalize(ex_feats_l + ex_feats_r, dim = -1) / self.args.feat_embed_dim
                ex_feats_norm = ex_feats.norm() / (ex_feats.size(0) * ex_feats.size(1))
                print(f"ex_feats_norm: {ex_feats_norm}")
                print(f"ex_feats mean: {ex_feats.mean().item()}")
                print(f"ex_feats abs mean: {ex_feats.abs().mean().item()}")

                self.layer0.weight = nn.Parameter(ex_feats / ex_feats_norm)
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                        
            if self.args.class_embed_dim == 1:
                correct_transform = torch.ones(1, 1, dtype = torch.float)
            else:
                correct_transform = torch.rand(1, self.args.class_embed_dim, dtype = torch.float)
                correct_transform = correct_transform - 0.5

            # print(f"correct_transform:\n{correct_transform}")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            # print(f"inv_correct_transform:\n{inv_correct_transform}")
            ex_correct = (ex_correct_base * 2 - 1) @ correct_transform

            # Do variance correction
            ex_correct_norm = ex_correct.norm() / (ex_correct.size(0) * ex_correct.size(1))
            inv_trans_norm = inv_correct_transform.norm() / (inv_correct_transform.size(0) * inv_correct_transform.size(1))
            
            print(f"ex_correct_norm: {ex_correct_norm}")
            print(f"ex_correct mean: {ex_correct.mean().item()}")
            print(f"ex_correct abs mean: {ex_correct.abs().mean().item()}")
            print()
            
            print(f"inv_trans_norm: {inv_trans_norm}")
            print(f"inv_correct_transform mean: {inv_correct_transform.mean().item()}")
            print(f"inv_correct_transform abs mean: {inv_correct_transform.abs().mean().item()}")
            # print(f"ex_feats_norm: {ex_feats_norm}")
            # print(f"ex_correct_norm: {ex_correct_norm}")
            # print(f"inv_trans_norm: {inv_trans_norm}")
            # if self.args.class_embed_dim == 1:
            #     inv_trans_norm = 1
            ex_correct = ex_correct / ex_correct_norm
            inv_correct_transform = inv_correct_transform / inv_trans_norm

            self.layer1.weight = nn.Parameter(ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))

            a, b = self.get_regression(ex_feats_l_base, ex_feats_r_base, ex_correct_base)
            a_sign = a / abs(a)
            a = abs(a)**(1/3)
            print(f"a: {a}, b: {b}")

            if self.args.which_ear == 'both':
                # print(f"self.layer0l.weight:\n{self.layer0l.weight}")
                # print(f"self.layer0l.weight x a:\n{self.layer0l.weight * a}")
                self.layer0l.weight = nn.Parameter(self.layer0l.weight * a)
                # print(f"new self.layer0l.weight:\n{self.layer0l.weight}")
                self.layer0r.weight = nn.Parameter(self.layer0r.weight * a)
            else:
                self.layer0.weight = nn.Parameter(self.layer0.weight * a)
            self.layer1.weight = nn.Parameter(self.layer1.weight * a)
            self.layer2.weight = nn.Parameter(self.layer2.weight * a * a_sign)
            self.layer2.bias = nn.Parameter(torch.tensor([b], dtype = torch.float))

            print(f"layer0.weight.size: {self.layer0.weight.size()}, layer1.weight.size: {self.layer1.weight.size()}, layer2.weight.size: {self.layer2.weight.size()}")
            print(f"self.layer0.weight: {self.layer0.weight.abs().mean().item()}, {self.layer0.weight.mean().item()}")
            print(f"self.layer1.weight: {self.layer1.weight.abs().mean().item()}, {self.layer1.weight.mean().item()}")
            print(f"self.layer2.weight: {self.layer2.weight.abs().mean().item()}, {self.layer2.weight.mean().item()}")
            # print(f"self.layer0.bias: {self.layer0.bias.norm()}")
            # print(f"self.layer1.bias: {self.layer1.bias.norm()}")
            # print(f"self.layer2.bias: {self.layer2.bias.norm()}")

            # quit()

        
        if initialisation == 'minerva3':
            ex_feats_l_base, ex_feats_r_base, ex_correct_base = inits
            ex_feats_l = ex_feats_l_base.detach()#) / self.args.feat_embed_dim
            ex_feats_r = ex_feats_r_base.detach()#) / self.args.feat_embed_dim
            # print(f"norm l mean: {norm_l.mean()}, min: {norm_l.min()}, max: {norm_l.max()}, len: {norm_l.size()}")
            # print(f"norm r mean: {norm_r.mean()}, min: {norm_r.min()}, max: {norm_r.max()}, len: {norm_r.size()}")
            # quit()

            # print(f"ex_feats_l_base:\n{ex_feats_l_base}")
            # print(f"ex_feats_r_base:\n{ex_feats_r_base}")
            # print(f"l_mean: {ex_feats_l_base.mean()}, r_mean: {ex_feats_r_base.mean()}")
            # print(f"l_max: {ex_feats_l_base.max()}, r_max: {ex_feats_r_base.max()}")
            # print(f"l_min: {ex_feats_l_base.min()}, r_min: {ex_feats_r_base.min()}")
            # print(f"l var: {torch.var(ex_feats_l_base)}, r var: {torch.var(ex_feats_r_base)}")
            # print()
            # print(f"l_mean: {ex_feats_l.mean()}, r_mean: {ex_feats_r.mean()}")
            # print(f"l_max: {ex_feats_l.max()}, r_max: {ex_feats_r.max()}")
            # print(f"l_min: {ex_feats_l.min()}, r_min: {ex_feats_r.min()}")
            # print(f"l var: {torch.var(ex_feats_l)}, r var: {torch.var(ex_feats_r)}")
            # print("dim 0 stats...")
            # print(f"l mean min/max: {ex_feats_l.mean(dim = 0).min()}, {ex_feats_l.mean(dim = 0).max()}")
            # print(f"r mean min/max: {ex_feats_r.mean(dim = 0).min()}, {ex_feats_r.mean(dim = 0).max()}")
            # print("dim 1 stats...")
            # print(f"l mean min/max: {ex_feats_l.mean(dim = 1).min()}, {ex_feats_l.mean(dim = 1).max()}")
            # print(f"r mean min/max: {ex_feats_r.mean(dim = 1).min()}, {ex_feats_r.mean(dim = 1).max()}")

            if self.args.which_ear == 'both':
                # norm_l = torch.linalg.vector_norm(ex_feats_l_base, dim = -1)
                # norm_r = torch.linalg.vector_norm(ex_feats_r_base, dim = -1)
                # print(f"norm_l dim: {norm_l.size()}")
                # print(f"norm_r dim: {norm_r.size()}")
                # norm_l = torch.pow(norm_l.unsqueeze(1), 2)
                # norm_r = torch.pow(norm_r.unsqueeze(1), 2)
                # print(f"norm_r dim: {norm_r.size()}")
                ex_feats_l = nn.functional.normalize(ex_feats_l / self.args.feat_embed_dim)
                ex_feats_r = nn.functional.normalize(ex_feats_r / self.args.feat_embed_dim)
                ex_feats_var = (torch.var(ex_feats_l) + torch.var(ex_feats_r)) / 2
                self.layer0l.weight = nn.Parameter(ex_feats_l)
                self.layer0r.weight = nn.Parameter(ex_feats_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
            else:
                if self.args.which_ear == 'left':
                    ex_feats = nn.functional.normalize(ex_feats_l, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'right':
                    ex_feats - nn.functional.normalize(ex_feats_r, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'mean':
                    ex_feats = nn.functional.normalize(ex_feats_l + ex_feats_r, dim = -1) / self.args.feat_embed_dim
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

            # print(f"correct_transform:\n{correct_transform}")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            # print(f"inv_correct_transform:\n{inv_correct_transform}")
            ex_correct = (ex_correct_base * 2 - 1) @ correct_transform

            # Do variance correction
            ex_correct_var = torch.var(ex_correct)
            inv_trans_var = torch.var(inv_correct_transform)
            if self.args.class_embed_dim == 1:
                inv_trans_var = 1
            ex_correct = ex_correct * (ex_feats_var / ex_correct_var)**0.5
            inv_correct_transform = inv_correct_transform * (inv_trans_var / ex_correct_var)**0.5

            self.layer1.weight = nn.Parameter(ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))

            a, b = self.get_regression(ex_feats_l_base, ex_feats_r_base, ex_correct_base)
            a_sign = a / abs(a)
            a = abs(a)**(1/3)
            print(f"a: {a}, b: {b}")

            if self.args.which_ear == 'both':
                print(f"self.layer0l.weight:\n{self.layer0l.weight}")
                print(f"self.layer0l.weight x a:\n{self.layer0l.weight * a}")
                self.layer0l.weight = nn.Parameter(self.layer0l.weight * a)
                print(f"new self.layer0l.weight:\n{self.layer0l.weight}")
                self.layer0r.weight = nn.Parameter(self.layer0r.weight * a)
            else:
                self.layer0.weight = nn.Parameter(self.layer0.weight * a)
            self.layer1.weight = nn.Parameter(self.layer1.weight * a)
            self.layer2.weight = nn.Parameter(self.layer2.weight * a * a_sign)
            self.layer2.bias = nn.Parameter(torch.tensor([b], dtype = torch.float))

            print(f"self.layer0.weight: {self.layer0.weight.abs().mean().item()}")
            print(f"self.layer1.weight: {self.layer1.weight.abs().mean().item()}")
            print(f"self.layer2.weight: {self.layer2.weight.abs().mean().item()}")
            print(f"self.layer0.bias: {self.layer0.bias.abs().mean().item()}")
            print(f"self.layer1.bias: {self.layer1.bias.abs().mean().item()}")
            print(f"self.layer2.bias: {self.layer2.bias.abs().mean().item()}")

            quit()
           

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

        output = {
            'logits': X,
            'preds': self.sigmoid(X)
        }
            
        return output
    

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




class minerva_transform(nn.Module):
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

        self.feat_embed_dim = args.feat_embed_dim
        self.class_embed_dim = args.class_embed_dim
        self.p_factor = args.p_factor
        self.device = args.device
        if ex_correct is not None:
            ex_correct = ex_correct * 2 - 1

        self.register_buffer('Dl', ex_feats_l)
        self.register_buffer('Dr', ex_feats_r)
        self.register_buffer('r', ex_correct)


        if self.feat_embed_dim is not None:
            # feat_embed_dim = args.feat_dim if args.feat_embed_dim is None else args.feat_embed_dim
            self.g = nn.Linear(in_features = args.feat_dim, out_features = self.feat_embed_dim)
        else:
            self.g = None
            
        if self.class_embed_dim is not None:
            self.h = nn.Linear(in_features = 1, out_features = self.class_embed_dim)
        else:
            self.h = None

        self.f = Minerva(args.p_factor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, D = None, r = None, left = False):

        if left:
            D = D if D is not None else self.Dl
        else:
            D = D if D is not None else self.Dr
        r = r if r is not None else self.r

        if self.g is not None:
            X = self.g(X)
            D = self.g(D)

        if self.h is not None:
            r = self.h(r) 

        echo = self.f(X, D, r)

        output = {
            'logits': echo,
            'preds': self.sigmoid(echo)
        }

        # echo = self.h(echo)
        # print(f"output:\n{echo}\n")

        return output







class ffnn_layers(nn.Module):
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
        if args.layer == -1:
            self.layer_weights = nn.Parameter(torch.ones(args.num_layers, dtype = torch.float) / args.num_layers)

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
            self.sm = nn.Softmax(dim = -1)
            

                
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

            
    def get_regression(self, feats_l, feats_r, targets):
        
        print(f"Starting weights regression...")
        
        self.to(self.device)
        # print(f"feats_l:\n{feats_l}")
        # print(f"feats_r:\n{feats_r}")
        
        x_l = self.forward(feats_l.to(self.device), left = True)
        # print(f"x_l:\n{x_l}") # dodgy
        x_r = self.forward(feats_r.to(self.device), left = False)
        # print(f"x_r:\n{x_r}") # ok
        # x_l_r = self.forward(feats_l.to(self.device), left = False)
        # print(f"x_l_r:\n{x_l_r}") # ok
        # x_r_l = self.forward(feats_r.to(self.device), left = True)
        # print(f"x_r_l:\n{x_r_l}") # dodgy
        # x_l_r = self.forward(feats_l.to(self.device), left = False)
        # x_r_l = self.forward(feats_r.to(self.device), left = True)
        # x_l_l = x_l_l['logits']
        # x_r_r = x_r_r['logits']
        # x_l_r = x_l_r['logits']
        # x_r_l = x_r_l['logits']
        # print(f"x_l_l mean: {x_l_l.mean().item()}, x_r_r mean: {x_r_r.mean().item()}")
        # print(f"x_l_r mean: {x_l_r.mean().item()}, x_r_l mean: {x_r_l.mean().item()}")
        # quit()
        x = torch.maximum(x_l['logits'], x_r['logits']).to('cpu').squeeze()
        targets = targets.squeeze()

        x_bar = x.mean().item()
        # print(f"x:\n{x}")
        # print(f"x_bar: {x_bar}")
        # print(f"layer0l.weight:\n{self.layer0l.weight}")
        # print(f"layer0r.weight:\n{self.layer0.weight}")
        if (x - x_bar).abs().max().item() == 0: # usually due to ReLU acting on EVERYTHING
            a0 = 1
        else:
            a0 = 1 / (x - x_bar).abs().max().item()
        b0 = -x_bar * a0
        # quit()

        # print(f"x:\n{x}")
        
        popt, _ = curve_fit(logit_func, x.detach().numpy(), targets.detach().numpy(), p0 = (a0, b0)) # , 
        a, b = popt
        print(f"corrections for intialisations - a: {a}, b: {b}")
        
        print(f"Finished weights regression.")
        return a, b

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
        self.eval()
        # ex_feats_l, ex_feats_r, ex_correct = inits

        
        if initialisation == 'minerva4':
            ex_feats_l_base, ex_feats_r_base, ex_correct_base = inits
            ex_feats_l = ex_feats_l_base.detach()
            ex_feats_r = ex_feats_r_base.detach()
            if self.args.which_ear == 'both':
                ex_feats_l = nn.functional.normalize(ex_feats_l)
                ex_feats_r = nn.functional.normalize(ex_feats_r)
                ex_feats_norm_l = ex_feats_l.norm() / (ex_feats_l.size(0) * ex_feats_l.size(1))
                ex_feats_norm_r = ex_feats_r.norm() / (ex_feats_r.size(0) * ex_feats_l.size(1))
                self.layer0l.weight = nn.Parameter(ex_feats_l / ex_feats_norm_l) 
                self.layer0r.weight = nn.Parameter(ex_feats_r / ex_feats_norm_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
            else:
                if self.args.which_ear == 'left':
                    ex_feats = nn.functional.normalize(ex_feats_l, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'right':
                    ex_feats - nn.functional.normalize(ex_feats_r, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'mean':
                    ex_feats = nn.functional.normalize(ex_feats_l + ex_feats_r, dim = -1) / self.args.feat_embed_dim
                ex_feats_norm = ex_feats.norm() / (ex_feats.size(0) * ex_feats.size(1))
                print(f"ex_feats_norm: {ex_feats_norm}")
                print(f"ex_feats mean: {ex_feats.mean().item()}")
                print(f"ex_feats abs mean: {ex_feats.abs().mean().item()}")

                self.layer0.weight = nn.Parameter(ex_feats / ex_feats_norm)
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                        
            if self.args.class_embed_dim == 1:
                correct_transform = torch.ones(1, 1, dtype = torch.float)
            else:
                correct_transform = torch.rand(1, self.args.class_embed_dim, dtype = torch.float)
                correct_transform = correct_transform - 0.5

            # print(f"correct_transform:\n{correct_transform}")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            # print(f"inv_correct_transform:\n{inv_correct_transform}")
            ex_correct = (ex_correct_base * 2 - 1) @ correct_transform

            # Do variance correction
            ex_correct_norm = ex_correct.norm() / (ex_correct.size(0) * ex_correct.size(1))
            inv_trans_norm = inv_correct_transform.norm() / (inv_correct_transform.size(0) * inv_correct_transform.size(1))
            
            print(f"ex_correct_norm: {ex_correct_norm}")
            print(f"ex_correct mean: {ex_correct.mean().item()}")
            print(f"ex_correct abs mean: {ex_correct.abs().mean().item()}")
            print()
            
            print(f"inv_trans_norm: {inv_trans_norm}")
            print(f"inv_correct_transform mean: {inv_correct_transform.mean().item()}")
            print(f"inv_correct_transform abs mean: {inv_correct_transform.abs().mean().item()}")
            # print(f"ex_feats_norm: {ex_feats_norm}")
            # print(f"ex_correct_norm: {ex_correct_norm}")
            # print(f"inv_trans_norm: {inv_trans_norm}")
            # if self.args.class_embed_dim == 1:
            #     inv_trans_norm = 1
            ex_correct = ex_correct / ex_correct_norm
            inv_correct_transform = inv_correct_transform / inv_trans_norm

            self.layer1.weight = nn.Parameter(ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))

            a, b = self.get_regression(ex_feats_l_base, ex_feats_r_base, ex_correct_base)
            a_sign = a / abs(a)
            a = abs(a)**(1/3)
            print(f"a: {a}, b: {b}")

            if self.args.which_ear == 'both':
                # print(f"self.layer0l.weight:\n{self.layer0l.weight}")
                # print(f"self.layer0l.weight x a:\n{self.layer0l.weight * a}")
                self.layer0l.weight = nn.Parameter(self.layer0l.weight * a)
                # print(f"new self.layer0l.weight:\n{self.layer0l.weight}")
                self.layer0r.weight = nn.Parameter(self.layer0r.weight * a)
            else:
                self.layer0.weight = nn.Parameter(self.layer0.weight * a)
            self.layer1.weight = nn.Parameter(self.layer1.weight * a)
            self.layer2.weight = nn.Parameter(self.layer2.weight * a * a_sign)
            self.layer2.bias = nn.Parameter(torch.tensor([b], dtype = torch.float))

            print(f"layer0.weight.size: {self.layer0.weight.size()}, layer1.weight.size: {self.layer1.weight.size()}, layer2.weight.size: {self.layer2.weight.size()}")
            print(f"self.layer0.weight: {self.layer0.weight.abs().mean().item()}, {self.layer0.weight.mean().item()}")
            print(f"self.layer1.weight: {self.layer1.weight.abs().mean().item()}, {self.layer1.weight.mean().item()}")
            print(f"self.layer2.weight: {self.layer2.weight.abs().mean().item()}, {self.layer2.weight.mean().item()}")
            # print(f"self.layer0.bias: {self.layer0.bias.norm()}")
            # print(f"self.layer1.bias: {self.layer1.bias.norm()}")
            # print(f"self.layer2.bias: {self.layer2.bias.norm()}")

            # quit()

        
        if initialisation == 'minerva3':
            ex_feats_l_base, ex_feats_r_base, ex_correct_base = inits
            ex_feats_l = ex_feats_l_base.detach()#) / self.args.feat_embed_dim
            ex_feats_r = ex_feats_r_base.detach()#) / self.args.feat_embed_dim
            # print(f"norm l mean: {norm_l.mean()}, min: {norm_l.min()}, max: {norm_l.max()}, len: {norm_l.size()}")
            # print(f"norm r mean: {norm_r.mean()}, min: {norm_r.min()}, max: {norm_r.max()}, len: {norm_r.size()}")
            # quit()

            # print(f"ex_feats_l_base:\n{ex_feats_l_base}")
            # print(f"ex_feats_r_base:\n{ex_feats_r_base}")
            # print(f"l_mean: {ex_feats_l_base.mean()}, r_mean: {ex_feats_r_base.mean()}")
            # print(f"l_max: {ex_feats_l_base.max()}, r_max: {ex_feats_r_base.max()}")
            # print(f"l_min: {ex_feats_l_base.min()}, r_min: {ex_feats_r_base.min()}")
            # print(f"l var: {torch.var(ex_feats_l_base)}, r var: {torch.var(ex_feats_r_base)}")
            # print()
            # print(f"l_mean: {ex_feats_l.mean()}, r_mean: {ex_feats_r.mean()}")
            # print(f"l_max: {ex_feats_l.max()}, r_max: {ex_feats_r.max()}")
            # print(f"l_min: {ex_feats_l.min()}, r_min: {ex_feats_r.min()}")
            # print(f"l var: {torch.var(ex_feats_l)}, r var: {torch.var(ex_feats_r)}")
            # print("dim 0 stats...")
            # print(f"l mean min/max: {ex_feats_l.mean(dim = 0).min()}, {ex_feats_l.mean(dim = 0).max()}")
            # print(f"r mean min/max: {ex_feats_r.mean(dim = 0).min()}, {ex_feats_r.mean(dim = 0).max()}")
            # print("dim 1 stats...")
            # print(f"l mean min/max: {ex_feats_l.mean(dim = 1).min()}, {ex_feats_l.mean(dim = 1).max()}")
            # print(f"r mean min/max: {ex_feats_r.mean(dim = 1).min()}, {ex_feats_r.mean(dim = 1).max()}")

            if self.args.which_ear == 'both':
                # norm_l = torch.linalg.vector_norm(ex_feats_l_base, dim = -1)
                # norm_r = torch.linalg.vector_norm(ex_feats_r_base, dim = -1)
                # print(f"norm_l dim: {norm_l.size()}")
                # print(f"norm_r dim: {norm_r.size()}")
                # norm_l = torch.pow(norm_l.unsqueeze(1), 2)
                # norm_r = torch.pow(norm_r.unsqueeze(1), 2)
                # print(f"norm_r dim: {norm_r.size()}")
                ex_feats_l = nn.functional.normalize(ex_feats_l / self.args.feat_embed_dim)
                ex_feats_r = nn.functional.normalize(ex_feats_r / self.args.feat_embed_dim)
                ex_feats_var = (torch.var(ex_feats_l) + torch.var(ex_feats_r)) / 2
                self.layer0l.weight = nn.Parameter(ex_feats_l)
                self.layer0r.weight = nn.Parameter(ex_feats_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
            else:
                if self.args.which_ear == 'left':
                    ex_feats = nn.functional.normalize(ex_feats_l, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'right':
                    ex_feats - nn.functional.normalize(ex_feats_r, dim = -1) / self.args.feat_embed_dim
                elif self.args.which_ear == 'mean':
                    ex_feats = nn.functional.normalize(ex_feats_l + ex_feats_r, dim = -1) / self.args.feat_embed_dim
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

            # print(f"correct_transform:\n{correct_transform}")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            # print(f"inv_correct_transform:\n{inv_correct_transform}")
            ex_correct = (ex_correct_base * 2 - 1) @ correct_transform

            # Do variance correction
            ex_correct_var = torch.var(ex_correct)
            inv_trans_var = torch.var(inv_correct_transform)
            if self.args.class_embed_dim == 1:
                inv_trans_var = 1
            ex_correct = ex_correct * (ex_feats_var / ex_correct_var)**0.5
            inv_correct_transform = inv_correct_transform * (inv_trans_var / ex_correct_var)**0.5

            self.layer1.weight = nn.Parameter(ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))

            a, b = self.get_regression(ex_feats_l_base, ex_feats_r_base, ex_correct_base)
            a_sign = a / abs(a)
            a = abs(a)**(1/3)
            print(f"a: {a}, b: {b}")

            if self.args.which_ear == 'both':
                print(f"self.layer0l.weight:\n{self.layer0l.weight}")
                print(f"self.layer0l.weight x a:\n{self.layer0l.weight * a}")
                self.layer0l.weight = nn.Parameter(self.layer0l.weight * a)
                print(f"new self.layer0l.weight:\n{self.layer0l.weight}")
                self.layer0r.weight = nn.Parameter(self.layer0r.weight * a)
            else:
                self.layer0.weight = nn.Parameter(self.layer0.weight * a)
            self.layer1.weight = nn.Parameter(self.layer1.weight * a)
            self.layer2.weight = nn.Parameter(self.layer2.weight * a * a_sign)
            self.layer2.bias = nn.Parameter(torch.tensor([b], dtype = torch.float))

            print(f"self.layer0.weight: {self.layer0.weight.abs().mean().item()}")
            print(f"self.layer1.weight: {self.layer1.weight.abs().mean().item()}")
            print(f"self.layer2.weight: {self.layer2.weight.abs().mean().item()}")
            print(f"self.layer0.bias: {self.layer0.bias.abs().mean().item()}")
            print(f"self.layer1.bias: {self.layer1.bias.abs().mean().item()}")
            print(f"self.layer2.bias: {self.layer2.bias.abs().mean().item()}")

            quit()
           

    def forward(self, X, left = False):
        # print(f"X.mean(0): {X.mean(dim = 0)}")
        # print(f"X.mean(1): {X.mean(dim = 1)}")
        # print(f"X.mean: {X.mean()}")

        if self.args.layer == -1:
            # X = X @ self.sm(self.layer_weights)
            X = X @ self.layer_weights
        
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

        output = {
            'logits': X,
            'preds': self.sigmoid(X)
        }
            
        return output
    

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



