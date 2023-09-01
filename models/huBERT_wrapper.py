from torch import Tensor, nn
import torch
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig, \
    WhisperModel, WhisperFeatureExtractor, WhisperForConditionalGeneration


class HuBERTWrapper_extractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        configuration = HubertConfig()
        model = HubertModel(configuration)
        self.model = model.feature_extractor
        print(self.model)

    def forward(self, data: Tensor):
        return self.model(data)


class HuBERTWrapper_full(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        configuration = HubertConfig()
        self.model = HubertModel(configuration)
        print(self.model)
        
    def forward(self, data: Tensor):
        
        my_output = self.model(data)
        return my_output[0]
    

class WhisperWrapper_encoder(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_feat_extractor = use_feat_extractor
        self.layer = layer

        if use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            model = WhisperModel.from_pretrained("openai/whisper-small")
        else:
            model = WhisperModel.from_pretrained(pretrained_model)

        self.model = model.encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, data):

        if self.use_feat_extractor:
            data = self.feature_extractor(data[0].to('cpu'), sampling_rate = 16000, return_tensors = 'pt')
            data = data.input_features.to(self.device)

        if self.layer is None:
            data = self.model(
                input_features = data, 
                return_dict = True
            )
            data = data[0]
        else:
            data = self.model(
                input_features = data, 
                return_dict = True,
                output_hidden_states = True
            )
            data = data.hidden_states[self.layer]

        return data
    
    
class WhisperWrapper_full(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, num_layers = 12, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # using layer = -1 returns all layers in form (1, time, feat_dim, layers)
        # otherwise single layer in form (1, time, feat_dim)

        self.num_layers = num_layers
        self.use_feat_extractor = use_feat_extractor
        if layer is None:
            self.layer = 12
        else:
            self.layer = layer

        if use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
    def forward(self, data):

        if self.use_feat_extractor:
            data = self.feature_extractor(data[0].to('cpu'), sampling_rate = 16000, return_tensors = 'pt')
            data = data.input_features.to(self.device)

        outputs = self.model.generate(
            input_features = data,
            output_hidden_states = True,
            return_dict_in_generate = True
        )

        if self.layer == -1:
            decoder_hidden = []
            for layer in range(self.num_layers):
                hidden = torch.stack([outputs.decoder_hidden_states[word][layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
                decoder_hidden.append(hidden.unsqueeze(0))
            decoder_hidden = torch.stack(decoder_hidden, dim = -1)
        else:
            decoder_hidden = torch.stack([outputs.decoder_hidden_states[word][self.layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
            decoder_hidden = decoder_hidden.unsqueeze(0)
        # print(f"decoder_hidden size: {decoder_hidden.size()}")

        # print(decoder_hidden.size())
        return decoder_hidden



class WhisperWrapperBase(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, num_layers = 6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # using layer = -1 returns all layers in form (1, time, feat_dim, layers)
        # otherwise single layer in form (1, time, feat_dim)

        self.num_layers = num_layers
        self.use_feat_extractor = use_feat_extractor
        if layer is None:
            self.layer = 12
        else:
            self.layer = layer

        if use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        if pretrained_model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)

        print(self.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
    def forward(self, data):

        if self.use_feat_extractor:
            data = self.feature_extractor(data[0].to('cpu'), sampling_rate = 16000, return_tensors = 'pt')
            data = data.input_features.to(self.device)

        outputs = self.model.generate(
            input_features = data,
            output_hidden_states = True,
            return_dict_in_generate = True
        )

        if self.layer == -1:
            decoder_hidden = []
            for layer in range(self.num_layers):
                hidden = torch.stack([outputs.decoder_hidden_states[word][layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
                decoder_hidden.append(hidden.unsqueeze(0))
            decoder_hidden = torch.stack(decoder_hidden, dim = -1)
        else:
            decoder_hidden = torch.stack([outputs.decoder_hidden_states[word][self.layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
            decoder_hidden = decoder_hidden.unsqueeze(0)
        # print(f"decoder_hidden size: {decoder_hidden.size()}")

        # print(decoder_hidden.size())
        return decoder_hidden