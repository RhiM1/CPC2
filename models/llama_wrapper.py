from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import torch
LLAMA_PATH = "/jmain02/home/J2AD003/txk68/gxc35-txk68/data/llama_checkpoints_hf/7B/"

class LlamaWrapper(nn.Module):
    def __init__(self):
        super(LlamaWrapper, self).__init__()
        self.llama = LlamaForCausalLM.from_pretrained(LLAMA_PATH).cuda()
        self.llama.requires_grad_(False)
        self.llama.eval()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(LLAMA_PATH)
        #self.max_pool = nn.MaxPool1d(4)

    def forward(self,x):
        x_tokens = self.tokenizer.encode(x, return_tensors="pt").cuda()
        out = self.llama.forward(x_tokens,output_hidden_states=True)
        last_hidden_state = out.hidden_states[-1]
        stacked_hidden_states = out.hidden_states[0]
        for h in out.hidden_states[1:-1]:
            stacked_hidden_states = torch.concat((stacked_hidden_states,h),dim=1)
        #stacked_hidden_states = self.max_pool(stacked_hidden_states)
        return stacked_hidden_states, last_hidden_state
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    model = LlamaWrapper()
    in_prompt = "The man in black fled across the desert, and the gunslinger followed."
    stacked,last = model(in_prompt)
    print(stacked.shape)
    print(last.shape)
    plt.figure(figsize=(20,20))
    stacked_to_plot = stacked.cpu().detach().numpy()[0].T
    stacked_to_plot_sig  = 1/(1+np.exp(-stacked_to_plot))
    plt.imshow(stacked_to_plot_sig,aspect="auto")
    plt.savefig("stacked.png")

