# Test hijack for transformer models
import torch.nn as nn 
import torch
import transformers 
from transformers import LlamaTokenizer, LlamaForCausalLM
import os 
os.environ['HF_DATASETS_OFFLINE'] = "1"
from datautils import get_dataset
import time
from quantize import sdxl_sequential, sdxl_pack
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file as safe_save
from pathlib import Path
import sys

def skip(*args, **kwargs):
	pass

# params

model = "DevaMalla/llama7b"
act_order = False
groupsize = 128
dataset = "ptb"
nsamples = 4
seed = 420
wbits = 4
true_sequential = True
sym = True
percdamp = 0.1


torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model).cuda().half()

model.seqlen = 2048
if act_order and groupsize != -1:
    raise ValueError('Cannot use act_order and groupsize together')

print("FP16 model response")
input_ids = tokenizer("Explain what protons and electrons are to me?", return_tensors="pt").to("cuda")
print(tokenizer.batch_decode(model.generate(**input_ids)))

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

print('Loading model...')
model.eval()

print('Loading data...')
dataloader = get_dataset(dataset, tokenizer, nsamples=nsamples, seed=seed, seqlen=1024)

print('Quantizing...')
tick = time.time()
quantizers = sdxl_sequential(model, dataloader, None, None, None, None, None, None, None, "cuda", wbits=wbits, nsamples=nsamples, true_sequential=true_sequential, sym=sym, percdamp=percdamp, groupsize=groupsize, act_order=act_order)
print(f"Total time: {time.time() - tick:.2f}s")

print('Packing...')
sdxl_pack(model, quantizers, wbits, groupsize)

sys.stdout = orig_stdout
f.close()

print("4Bit model response")
input_ids = tokenizer("Explain what protons and electrons are to me?", return_tensors="pt").to("cuda")
print(tokenizer.batch_decode(model.generate(**input_ids)))

print('Done.')