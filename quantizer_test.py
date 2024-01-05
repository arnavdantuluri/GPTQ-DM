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

print(gptq)
print(quantizers)

'''
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
0 self_attn.q_proj
1 self_attn.k_proj
2 self_attn.v_proj
3 self_attn.o_proj
4 mlp.gate_proj
5 mlp.up_proj
6 mlp.down_proj
{'self_attn.q_proj': <gptq.GPTQ object at 0x1534016c3d30>, 'self_attn.k_proj': <gptq.GPTQ object at 0x1534016c3df0>, 'self_attn.v_proj': <gptq.GPTQ object at 0x153428576aa0>, 'self_attn.o_proj': <gptq.GPTQ object at 0x1534016c37c0>, 'mlp.gate_proj': <gptq.GPTQ object at 0x1534016c2f20>, 'mlp.up_proj': <gptq.GPTQ object at 0x1534016c1e70>, 'mlp.down_proj': <gptq.GPTQ object at 0x1534016c32e0>}
{'model.layers.0.self_attn.q_proj': (Quantizer(), 0, 0), 'model.layers.1.self_attn.k_proj': (Quantizer(), 0, 0), 'model.layers.2.self_attn.v_proj': (Quantizer(), 0, 0), 'model.layers.3.self_attn.o_proj': (Quantizer(), 0, 0), 'model.layers.4.mlp.gate_proj': (Quantizer(), 0, 0), 'model.layers.5.mlp.up_proj': (Quantizer(), 0, 0), 'model.layers.6.mlp.down_proj': (Quantizer(), 0, 0)}
'''