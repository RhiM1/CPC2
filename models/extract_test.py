import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from fairseq.models.hubert.hubert import HubertModel
x = torch.rand(1, 3, 224, 224)

model = resnet50()
print(model)
return_nodes = {
    "layer4.2.relu_2": "layer4"
}

model2 = create_feature_extractor(model, return_nodes=return_nodes)
intermediate_outputs = model2(x)



print(intermediate_outputs['layer4'].shape)


