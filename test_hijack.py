# Test hijack for transformer models
import torch.nn as nn 
import torch
import transformers 
from transformers import AutoTokenizer, LlamaForCausalLM
from datautils import get_dataset
from time import time
from quantize import sdxl_sequential, sdxl_pack

def skip(*args, **kwargs):
	pass

# params

model = "huggyllama/llama-7b"
act_order = True
groupsize = 128
dataset = "ptb"
nsamples = 128
seed = 42
wbits = 4
true_sequential = True
sym = True
percdamp = 0.1


torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
model.seqlen = 2048
if act_order and groupsize != -1:
    raise ValueError('Cannot use act_order and groupsize together')

print("FP16 model response")
model.generate()

print('Loading model...')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

print('Loading data...')
dataloader = get_dataset(dataset, tokenizer, nsamples=nsamples, seed=seed, seqlen=model.seqlen)

print('Quantizing...')
tick = time.time()
quantizers = sdxl_sequential(model, dataloader, None, None, None, None, None, None, None, "cuda", wbits=wbits, nsamples=nsamples, true_sequential=true_sequential, sym=sym, percdamp=percdamp, groupsize=groupsize, act_order=act_order)
print(f"Total time: {time.time() - tick:.2f}s")

print('Packing...')
sdxl_pack(model, quantizers, wbits, groupsize)

print('Done.')