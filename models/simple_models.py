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

        if initialisation == 'minerva':
            ex_feats_l, ex_feats_r, ex_correct = inits
            # print(f"ex_correct:\n{ex_correct}\n")
            if self.args.which_ear == 'both':
                self.layer0l.weight = nn.Parameter(nn.functional.normalize(ex_feats_l, dim = -1))
                self.layer0r.weight = nn.Parameter(nn.functional.normalize(ex_feats_r, dim = -1))
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                
            else:
                if self.args.which_ear == 'left':
                    ex_feats = ex_feats_l
                elif self.args.which_ear == 'right':
                    ex_feats = ex_feats_r
                self.layer0.weight = nn.Parameter(nn.functional.normalize(ex_feats, dim = -1))
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))

            correct_transform = torch.ones(1, self.args.class_embed_dim, dtype = torch.float)
            if self.args.class_embed_dim > 1:
                nn.init.kaiming_uniform_(correct_transform, a=math.sqrt(5))
            # print(f"correct_transform:\n{correct_transform}\n")
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            ex_correct = (ex_correct * 2 - 1) @ correct_transform
            # print(f"ex_correct transformed:\n{ex_correct}\n")

            self.layer1.weight = nn.Parameter(ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))
            
            # test_ex_correct = self.layer2(ex_correct)
            # print(f"test_ex_correct: \n{test_ex_correct}")
            # quit()


        if initialisation == 'minerva_scaled':
            ex_feats_l_base, ex_feats_r_base, ex_correct_base = inits
            # print(f"ex_correct:\n{ex_correct}\n")
            if self.args.which_ear == 'both':
                ex_feats_l = nn.functional.normalize(ex_feats_l_base, dim = -1)
                ex_feats_r = nn.functional.normalize(ex_feats_r_base, dim = -1)
                scale0_l = ex_feats_l_base @ ex_feats_l.t()
                scale0_r = ex_feats_r_base @ ex_feats_r.t()
                scale0_l = (scale0_l.sum() - torch.trace(scale0_l)) / (self.args.feat_embed_dim - 1)**2
                scale0_r = (scale0_r.sum() - torch.trace(scale0_r)) / (self.args.feat_embed_dim - 1)**2
                self.layer0l.weight = nn.Parameter(ex_feats_l / scale0_l)
                self.layer0r.weight = nn.Parameter(ex_feats_r / scale0_r)
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
                self.layer0.weight = nn.Parameter(ex_feats / scale0)
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

            self.layer1.weight = nn.Parameter(ex_correct.t() / scale1)
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            X = self.layer1(X)

            scale2 = (X @ inv_correct_transform.t()).mean()

            self.layer2.weight = nn.Parameter(inv_correct_transform / scale2)
            self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))



        elif initialisation == 'minerva_scaled3':
            ex_feats_l, ex_feats_r, ex_correct = inits
            ex_correct_temp = ex_correct

            if self.args.which_ear == 'both':
                ex_feats_l = nn.functional.normalize(ex_feats_l, dim = -1)
                ex_feats_r = nn.functional.normalize(ex_feats_r, dim = -1)
                ex_feats_var = (ex_feats_l.var() + ex_feats_r.var()) / 2
                ex_feats_l = self.args.p_factor * ex_feats_l
                ex_feats_r = self.args.p_factor * ex_feats_r
                ex_feats_mean = (ex_feats_l.mean() + ex_feats_r.mean()) / 2
                self.layer0l.weight = nn.Parameter(ex_feats_l)
                self.layer0r.weight = nn.Parameter(ex_feats_r)
                self.layer0l.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                self.layer0r.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
            else:
                if self.args.which_ear == 'left':
                    ex_feats = nn.functional.normalize(ex_feats_l, dim = -1)
                    ex_feats_var = ex_feats_l.var()
                    ex_feats_mean = ex_feats_l.mean()
                elif self.args.which_ear == 'right':
                    ex_feats = nn.functional.normalize(ex_feats_r, dim = -1)
                    ex_feats_var = ex_feats_r.var()
                    ex_feats_mean = ex_feats_r.mean()
                ex_feats = self.args.p_factor * ex_feats
                self.layer0.weight = nn.Parameter(ex_feats)
                self.layer0.bias = nn.Parameter(torch.zeros(self.args.feat_embed_dim))
                

            
            correct_transform = torch.ones(1, self.args.class_embed_dim, dtype = torch.float)
            if self.args.class_embed_dim > 1:
                nn.init.kaiming_uniform_(correct_transform, a=math.sqrt(5))
            # print(f"correct_transform:\n{correct_transform}\n")
            correct_transform_var = correct_transform.var()
            correct_transform = correct_transform / correct_transform_var**0.5
            inv_correct_transform = 1 / (self.args.class_embed_dim * correct_transform)
            inv_correct_transform_var = inv_correct_transform.var()
            print(f"ex_feats mean: {ex_feats_mean}")
            # print(f"correct_transform:\n{correct_transform}")
            # print(f"inv_correct_transform:\n{inv_correct_transform}")
            print(f"correct_transform mean: {correct_transform.mean()}")
            print(f"inv_correct_transform mean: {inv_correct_transform.mean()}")
            print(f"ex_feats_var: {ex_feats_var}")
            print(f"correct_transform_var: {correct_transform_var}")
            print(f"inv_correct_transform_var: {inv_correct_transform_var}")
            # correct_transform = correct_transform * ((inv_correct_transform_var / correct_transform_var)/2)**0.5
            # inv_correct_transform = correct_transform * ((correct_transform_var / inv_correct_transform_var)/2)**0.5

            # inv_correct_transform_var = inv_correct_transform.var()
            # correct_transform_var = correct_transform.var()
            # print(f"correct_transform_var: {correct_transform_var}")
            # print(f"inv_correct_transform_var: {inv_correct_transform_var}")
            # quit()

            ex_correct = (ex_correct * 2 - 1) @ correct_transform
            ex_correct_var = ex_correct.var()
            # print(f"ex_correct_var: {ex_correct_var}")

            # print(f"ex_correct temp:\n{ex_correct_temp}")
            
            ex_correct = self.args.p_factor * ex_correct * (ex_feats_var / ex_correct_var)**0.5
            # print(f"ex_correct transformed:\n{ex_correct}\n")
            # print(f"ex_correct:\n{ex_correct}")

            self.layer1.weight = nn.Parameter(ex_correct.t())
            self.layer1.bias = nn.Parameter(torch.zeros(self.args.class_embed_dim, dtype = torch.float))

            self.layer2.weight = nn.Parameter(inv_correct_transform)
            self.layer2.bias = nn.Parameter(torch.tensor([-1], dtype = torch.float))


            # print(f"corrected ex_correct:\n{self.layer2(ex_correct) / (ex_feats_var / ex_correct_var)**0.5}")

            # quit()
            
            # self.layer1.weight = nn.Parameter(ex_correct.t())
            # self.layer1.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))

            # self.layer2.weight = nn.Parameter(torch.ones(1, 1, dtype = torch.float))
            # self.layer2.bias = nn.Parameter(torch.zeros(1, dtype = torch.float))


    def forward(self, X, left = False):
        
        # print(f"X:\n{X}")
        if self.args.normalize:
            X = nn.functional.normalize(X, dim = -1)
        if self.args.use_layer_norm:
            X = self.ln0(X)
        if self.args.which_ear != 'both':
            X = self.layer0(X)
        elif left:
            X = self.layer0l(X)
        else:
            X = self.layer0r(X)
        X = self.do0(X)
        X = self.activation0(X)
        # print(f"X after L0 activation:\n{X}")
        if self.args.use_layer_norm:
            X = self.ln1(X)
        X = self.layer1(X)
        X = self.do1(X)
        X = self.activation1(X)
        # print(f"X after L1 activation:\n{X}")
        if self.args.use_layer_norm:
            X = self.ln2(X)
        X = self.layer2(X)
        # print(X)
        # print(self.sigmoid(X))
        # print(f"X after L2 activation:\n{X}")
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






