import torch
import torch.nn.functional as F
from torch import nn
from speechbrain.processing.features import spectral_magnitude,STFT
import os

from models.ni_predictors import PoolAttFF


class base_model(nn.Module):

    def __init__(
            self, 
            config = None, 
            load_dir = None
        ):
        super(base_model, self).__init__()

        if load_dir is not None:
            self.load_pretrained(load_dir = load_dir)
        elif config is None:
            print("Must provide either save location or config file for loading model")
        else:
            self.config = config
        self.ex_feats = None
        self.ex_targets = None

    def save_pretrained(self, output_dir):    
        torch.save(self.config, output_dir + "/config.json")
        torch.save(self.state_dict, output_dir + "/model.mod")
        if self.ex_feats is not None:
            torch.save(self.ex_feats.to('cpu'), output_dir + "/ex_feats.pt")
        if self.ex_targets is not None:
            torch.save(self.ex_targets.to('cpu'), output_dir + "/ex_targets.pt")
             

    def load_pretrained(self, load_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = torch.load(load_dir + "/config.json")
        state_dict = torch.load(load_dir + "/model.mod")
        self.load_state_dict(state_dict)
        if os.path.exists(load_dir + "/ex_feats.pt"):
            self.ex_feats = torch.load(load_dir + "/ex_feats.pt")
        if os.path.exists(load_dir + "/ex_targets.pt"):
            self.ex_targets = torch.load(load_dir + "/ex_targets.pt")
        

class minerva(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        ex_targets = None,
        ex_features_l = None,
        ex_features_r = None,
        ex_IDX = None,
        config = None,
        load_dir = None
    ):
        super().__init__(config = config, load_dir = load_dir)
        
        if self.config['feat_dim'] == None:
            self.config['feat_dim'] = self.config['input_dim']

        print(f"config:\n{self.config}")

        # if config['use_g']:
        #     self.g = nn.Linear(
        #         in_features = self.config['input_dim'],
        #         out_features = self.config['feat_dim'],
        #         bias = False
        #     )
        #     print(f"g:\n{self.g}")

        self.g = nn.LSTM(
            input_size = self.config['input_dim'],
            hidden_size = self.config['feat_dim'],
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        

        if config['dropout'] > 0:
            self.do = nn.Dropout(p = config['dropout'])

        self.ex_features_l = ex_features_l
        self.ex_features_r = ex_features_r
        self.ex_targets = ex_targets
        self.ex_idx = ex_IDX
        
        if self.config['train_ex_features']:
            self.add_ex_feats_l = nn.Parameter(torch.zeros_like(self.ex_features_l.data))
            self.add_ex_feats_r = nn.Parameter(torch.zeros_like(self.ex_features_r.data))
        if self.config['train_ex_class'] and self.ex_targets is not None:
            self.add_ex_targets = nn.Parameter(torch.zeros_like(self.ex_targets))
        
        # self.set_exemplars(ex_features, ex_targets, ex_IDX)
        # self.initialise_exemplars()

    def set_exemplars(self, ex_features_l, ex_features_r, ex_targets, ex_IDX):
        
        if ex_targets is None:
            self.ex_targets = None
        else:
            self.ex_targets = nn.Parameter(ex_targets, requires_grad = False)
            print(f"ex_targets size:: \n{self.ex_targets.size()}")
            if self.config['train_ex_class']:
                self.add_ex_targets = nn.Parameter(torch.zeros_like(self.ex_targets))
                print(f"add_ex_targets size:: \n{self.add_ex_targets.size()}")


        if ex_features_l is None:
            self.ex_features_l = None
        else:
            ex_features_l, len_ex_features_l = self.unpack_exemplars(ex_features_l)
            self.ex_features_l = nn.Parameter(ex_features_l, requires_grad = False)
            self.len_ex_features_l = nn.Parameter(len_ex_features_l, requires_grad = False)
            print(f"ex_features size:: \n{self.ex_features_l.size()}")
            if self.config['train_ex_features']:
                self.add_ex_feats_l = nn.Parameter(torch.zeros_like(self.ex_features_l.data))
                print(f"add_ex_feats size:: \n{self.add_ex_feats_l.size()}")

        if ex_features_r is None:
            self.ex_features_r = None
        else:
            ex_features_r, len_ex_features_r = self.unpack_exemplars(ex_features_r)
            self.ex_features_r = nn.Parameter(ex_features_r, requires_grad = False)
            self.len_ex_features_r = nn.Parameter(len_ex_features_r, requires_grad = False)
            print(f"ex_features size:: \n{self.ex_features_r.size()}")
            if self.config['train_ex_features']:
                self.add_ex_feats_r = nn.Parameter(torch.zeros_like(self.ex_features_r.data))
                print(f"add_ex_feats size:: \n{self.add_ex_feats_r.size()}")

        if ex_IDX is None:
            self.ex_IDX = None
        else:
            self.ex_IDX = nn.Parameter(ex_IDX, requires_grad = False)
            print(f"ex_IDX size: \n{self.ex_IDX.size()}")


    def forward(self, features_l, fatures_r, ex_features_l = None, ex_features_r = None, ex_targets = None, p_factor = None):
        
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, phone_dim)
        # ex_reps has dim (num_classes, phone_dim)
        # print(f"features.size: {features.size()}, ex_features.size: {self.ex_features.size()}")

        if p_factor is None:
            p_factor = self.config['p_factor']

        if ex_features_l is None:
            ex_features_l = self.ex_features_l
        if ex_features_r is None:
            ex_features_r = self.ex_features_r
        if self.config['train_ex_feats']:
            ex_features_l += self.add_ex_feats_l
            ex_features_r += self.add_ex_feats_r
        features_l = self.g(features_l)
        features_r = self.g(features_r)
        ex_features_l = self.g(ex_features_l)
        ex_features_r = self.g(ex_features_r)
        if self.config['dropout'] > 0:
            features_l = self.do(features_l)
            features_r = self.do(features_r)
            ex_features_l = self.do(ex_features_l)
            ex_features_r = self.do(ex_features_r)

        if ex_targets is None:
            ex_targets = self.ex_targets
        if self.config['train_ex_class']:
            ex_targets += self.add_ex_targets

        # s has dim (batch_size, ex_batch_size)
        s_l = torch.matmul(
            nn.functional.normalize(features_l, dim = 1), 
            torch.t(nn.functional.normalize(ex_features_l, dim = 1))
        )
        s_r = torch.matmul(
            nn.functional.normalize(features_r, dim = 1), 
            torch.t(nn.functional.normalize(ex_features_r, dim = 1))
        )

        # a has dim (batch_size, ex_batch_size)
        a_l = self.activation(s_l, p_factor)
        a_l = nn.functional.normalize(a_l, dim = 1, p = 1)
        a_r = self.activation(s_r, p_factor)
        a_r = nn.functional.normalize(a_r, dim = 1, p = 1)

        # echo has dim (batch_size, phone_dim)
        echo_l = torch.matmul(a_l, ex_targets)
        echo_r = torch.matmul(a_r, ex_targets)

        return echo_l, echo_r
    
    def unpack_exemplars(self, padded_ex_feats):

        ex_feats = padded_ex_feats.data
        len_ex_feats = padded_ex_feats.lengths
        len_ex_feats = (len_ex_feats * ex_feats.size(1)).to(torch.int)
        
        return ex_feats, len_ex_feats
    
    def pack_exemplars(self, ex_feats, len_ex_feats):

        packed_ex_feats = torch.nn.utils.rnn.pack_padded_sequence(ex_feats, len_ex_feats, batch_first=True)

        return packed_ex_feats

    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.config['p_factor']

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))



class ExemplarMetricPredictor(nn.Module):
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
        self, minerva_config, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim=512, 
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

        self.minerva = minerva(config = minerva_config)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        out,_ = self.blstm(x)
        print(f"out size: {out.size()}")
        #out = out_feats
        out = self.attenPool(out)
        out = self.minerva(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,_


class ExemplarMetricPredictorCombo(nn.Module):
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
        self, minerva_config, dim_extractor=1024, hidden_size=1024//2, activation=nn.LeakyReLU,
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

        self.minerva = minerva(config = minerva_config)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats_full, feats_extact):

        out_full,_ = self.blstm_last(feats_full)
        out_extract,_ = self.blstm_encoder(feats_extact)

        #print(out_full.shape,out_extract.shape)
        out = torch.cat((out_full,out_extract),dim=2)
        out = self.attenPool(out)
        out = self.minerva(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,_


