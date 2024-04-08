from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoConfig
from gptq import GPTQ, Quantizer
# params

model = "DevaMalla/llama7b"
act_order = False
groupsize = 128
dataset = "ptb"
nsamples = 128
seed = 42
wbits = 4
true_sequential = True
sym = True
percdamp = 0.1

config = AutoConfig.from_pretrained(model)
layer = LlamaDecoderLayer(config, 0)

quantizers = {}
full = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}

sequential = [list(full.keys())]
outs = []
idx = 0
# For each subset of linear layers
for names in sequential:
    subset = {n: full[n] for n in names}
    gptq = {}

    # Prepare a quantizer for each linear layer
    for name in subset:
        print(idx)
        gptq[idx] = GPTQ(layer) # subset[name] -> linear layer
        gptq[idx].quantizer = Quantizer()
        gptq[idx].quantizer.configure(4, perchannel=True, sym=True, mse=False)

    # With the data collected, quantize the layers
    for i, name in enumerate(subset):
        print(i, name)
        quantizers['model.layers.%d' % (i)] = (gptq[idx].quantizer, 0, 0)
        gptq[idx].free()

    idx += 1