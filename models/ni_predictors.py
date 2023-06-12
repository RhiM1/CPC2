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

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, dim_head_in, out_dim = 1):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, 2*dim_head_in)
        self.linear2 = nn.Linear(2*dim_head_in, 1)
        
        self.linear3 = nn.Linear(dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x  

class LlamaMetricPredictor(nn.Module):
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

        self.llama = LlamaWrapper()
        self.max_pool = nn.MaxPool1d(4)
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        out_feats,_ = self.llama(x)
        out_feats = self.max_pool(out_feats)
        #print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class LlamaHuBERTMetricPredictor(nn.Module):
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
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)
        #self.BN_1 = nn.BatchNorm2d(512)
        #self.BN_2 = nn.(512)
        self.llama = LlamaWrapper()
        self.huBERT = HuBERTWrapper_extractor()
        
        self.llama.requires_grad_(False)
        self.huBERT.requires_grad_(False)

        self.max_pool = nn.MaxPool1d(8)
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,aud_feats):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)

        out_feats,_ = self.llama(x)
        
        aud = self.huBERT(aud_feats).permute(0,2,1)
        out_feats = self.max_pool(out_feats)
        #out_feats = self.BN_1(out_feats)
        #aud = self.BN_2(aud)
        print(out_feats.shape,aud.shape)
        out_feats = torch.cat((out_feats,aud),dim=1)
        print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class SpecMetricPredictor(nn.Module):
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
        self, dim_extractor=257, hidden_size=257//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        
        
        self.stft = STFT(hop_length=16,win_length=32,sample_rate=16000,n_fft=512,window_fn=torch.hamming_window)

        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor-1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        feats = self.stft(x)
        feats = spectral_magnitude(feats, power=0.5)
        out_feats = torch.log1p(feats)
        #print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class SpecMetricPredictorBig(nn.Module):
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
        self, dim_extractor=513, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        
        
        self.stft = STFT(hop_length=16,win_length=32,sample_rate=16000,n_fft=1024,window_fn=torch.hamming_window)

        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor-1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        feats = self.stft(x)
        feats = spectral_magnitude(feats, power=0.5)
        out_feats = torch.log1p(feats)
        #print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class XLSRMetricPredictorCombo(nn.Module):
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


        self.feat_extract = Wav2Vec2Wrapper_no_helper()
        self.feat_extract.requires_grad_(False)

        
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

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        mod_out = self.feat_extract(x)
        out_feats_full =  mod_out['last_hidden_state']#.permute(0,2,1)
        out_feats_extact = mod_out['extract_features']
        out_full,_ = self.blstm_last(out_feats_full)
        out_extract,_ = self.blstm_encoder(out_feats_extact)

        #print(out_full.shape,out_extract.shape)
        out = torch.cat((out_full,out_extract),dim=2)
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats_full




class XLSRMetricPredictorFull(nn.Module):
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


        self.feat_extract = Wav2Vec2Wrapper_no_helper()
        self.feat_extract.requires_grad_(False)

        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x)['last_hidden_state']#.permute(0,2,1)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class XLSRMetricPredictorEncoder(nn.Module):
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
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)


        self.feat_extract = Wav2Vec2Wrapper_encoder_only().eval()
        self.feat_extract.requires_grad_(False)
        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x).permute(0,2,1)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

        
class HuBERTMetricPredictorFull(nn.Module):
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
        self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)


        self.feat_extract = HuBERTWrapper_full()
        self.feat_extract.requires_grad_(False)

        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x)#.permute(0,2,1)
        out,_ = self.blstm(out_feats)
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class HuBERTMetricPredictorEncoder(nn.Module):
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
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        #self.BN = nn.BatchNorm1d(num_features=1, momentum=0.01)


        self.feat_extract = HuBERTWrapper_extractor()
        self.feat_extract.requires_grad_(False)

        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x).permute(0,2,1)
        #print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

if __name__ == "__main__":
    print("Testing the model")
    model = LlamaMetricPredictor().cuda()
    x = "the quick brown fox jumped over the lazy dog"
    y,_ = model(x)
    print(y)
    print(y.shape)