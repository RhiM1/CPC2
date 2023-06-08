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

    def save_pretrained(self, output_dir):    
        torch.save(self.config, output_dir + "/config.json")
        torch.save(self.state_dict, output_dir + "/model.mod")
             

    def load_pretrained(self, load_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = torch.load(load_dir + "/config.json")
        state_dict = torch.load(load_dir + "/model.mod")
        self.load_state_dict(state_dict)
        

class minerva(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        ex_targets = None,
        ex_features = None,
        ex_IDX = None,
        config = None,
        load_dir = None
    ):
        super().__init__(config = config, load_dir = load_dir)
        
        if self.config['feat_dim'] == None:
            self.config['feat_dim'] = self.config['input_dim']

        print(f"config:\n{self.config}")

        if config['use_g']:
            self.g = nn.Linear(
                in_features = self.config['input_dim'],
                out_features = self.config['feat_dim'],
                bias = False
            )
            print(f"g:\n{self.g}")

        if config['dropout'] > 0:
            self.do = nn.Dropout(p = config['dropout'])
        self.set_exemplars(ex_features, ex_targets, ex_IDX)
        self.initialise_exemplars()

    def set_exemplars(self, ex_features, ex_classes, ex_IDX):
        
        self.ex_classes = nn.Parameter(ex_classes, requires_grad = False)
        print(f"ex_classes init: \n{self.ex_classes}")
        if ex_features is None:
            self.ex_features = None
        else:
            self.ex_features = nn.Parameter(ex_features, requires_grad = False)
            print(f"ex_features init: \n{self.ex_features}")
        if ex_IDX is None:
            self.ex_IDX = None
        else:
            self.ex_IDX = nn.Parameter(ex_IDX, requires_grad = False)
            print(f"ex_IDX init: \n{self.ex_IDX}")


    def initialise_exemplars(self):

        if self.config['class_dim'] is None:
            self.class_reps = torch.nn.Parameter(
                nn.functional.one_hot(torch.arange(self.config['num_labels'])).type(torch.float),
                requires_grad = self.config['train_class_reps']
            )
            if self.config['train_ex_class']:
                self.add_ex_class_reps = torch.nn.Parameter(
                    torch.zeros(len(self.ex_classes), self.config['num_labels'], dtype = torch.float)
                )
        else:
            self.class_reps = torch.nn.Parameter(
                torch.rand(self.config['num_labels'], self.config['class_dim'], dtype = torch.float) * 2 - 1,
                requires_grad = self.config['train_class_reps']
            )
            if self.config['train_ex_class']:
                self.add_ex_class_reps = torch.nn.Parameter(
                    torch.zeros(len(self.ex_classes), self.config['class_dim'], dtype = torch.float)
                )

        # print(f"class_reps init: \n{self.class_reps}")
        # if self.config['train_ex_class']:
        #     print(f"add_ex_class_reps init: \n{self.add_ex_class_reps}")
        if self.config['train_ex_feats']:
            self.add_feats = nn.Parameter(
                torch.zeros(len(self.ex_classes), self.config['feat_dim'], dtype = torch.float)
            )
        
        print()
        if self.ex_features is not None:
            print("ex_features.size:", self.ex_features.size())
        print("ex_classes.size:", self.ex_classes.size())
        print("class_reps.size:", self.class_reps.size())
        print()


    def forward(self, features, ex_features = None, p_factor = None):
        
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, phone_dim)
        # ex_reps has dim (num_classes, phone_dim)
        # print(f"features.size: {features.size()}, ex_features.size: {self.ex_features.size()}")

        if ex_features is None:
            ex_features = self.ex_features

        if p_factor is None:
            p_factor = self.config['p_factor']

        if self.config['use_g']:
            features = self.g(features)
            ex_features = self.g(ex_features)

        if self.config['train_ex_feats']:
            ex_features += self.add_feats

        if self.config['dropout'] > 0:
            features = self.do(features)
            ex_features = self.do(ex_features)

        ex_class_reps = torch.matmul(
            torch.nn.functional.normalize(self.ex_classes, p = 1, dim = -1),
            self.class_reps
        )
        if self.config['train_ex_class']:
            ex_class_reps += self.add_ex_class_reps

        # s has dim (batch_size, ex_batch_size)
        s = torch.matmul(
            nn.functional.normalize(features, dim = 1), 
            torch.t(nn.functional.normalize(ex_features, dim = 1))
        )

        # a has dim (batch_size, ex_batch_size)
        a = self.activation(s, p_factor)
        a = nn.functional.normalize(a, dim = 1, p = 1)

        # echo has dim (batch_size, phone_dim)
        echo = torch.matmul(a, ex_class_reps)

        return echo

    
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
        #out = out_feats
        out = self.attenPool(out)
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


