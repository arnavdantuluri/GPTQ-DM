# Modified from: https://github.com/fpgaminer/GPTQ-triton
# Quantize a model using the GPTQ algorithm.
import argparse
import json
import library.train_util as train_util
import library.sdxl_train_util as sdxl_train_util
from pathlib import Path
import shutil
import time
from typing import Optional

import torch
from torch.fx import symbolic_trace
import torch.nn as nn
from collections import namedtuple
from datautils import get_dataset
from gptq import GPTQ, Quantizer
import gptq
from quant_linear import QuantLinear
import quant_linear
from tqdm import tqdm
from transformers import AutoTokenizer
from diffusers import DiffusionPipeline
from safetensors.torch import save_file as safe_save
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from diffusers import DDPMScheduler
'''
sample = torch.rand(2, 4, 128, 128).cuda().half()
timesteps = torch.rand([]).cuda()
encoder_hidden_states = torch.rand(2, 77, 2048).cuda().half()
added_cond_kwargs = {
    'text_embeds': torch.rand(2, 1280).cuda().half(),
    'time_ids': torch.rand(2, 6).cuda().half(),
}

'''
noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )

def main(model, dataloader, act_order, groupsize, wbits, nsamples, true_sequential, percdamp, sym, path, safetensors: bool):
	save = Path(path)

	if act_order and groupsize != -1:
		raise ValueError('Cannot use act_order and groupsize together')

	print('Loading model...')
	model = get_sdxl(model)
	model.eval()

	print('Quantizing...')
	tick = time.time()
	quantizers = sdxl_sequential(model, dataloader, device='cuda', wbits=wbits, nsamples=nsamples, sym=sym, percdamp=percdamp, groupsize=groupsize, act_order=act_order)
	print(f"Total quantizing time: {time.time() - tick:.2f}s")

	print('Packing...')
	tick = time.time()
	sdxl_pack(model, quantizers, wbits, groupsize)
	print(f"Total packing time: {time.time() - tick:.2f}s")

	print('Saving...')
	save.mkdir(parents=True, exist_ok=True)

	# Save the model
	if safetensors:
		safe_save(model.state_dict(), save / 'model.safetensors')
	else:
		torch.save(model.state_dict(), save / 'model.pt')
	
	print('Done.')


def get_sdxl(model):
	def skip(*args, **kwargs):
		pass

	# NOTE: This is a nasty hack, but it speeds up model building by a huge amount
	torch.nn.init.kaiming_uniform_ = skip
	torch.nn.init.uniform_ = skip
	torch.nn.init.normal_ = skip

	model = DiffusionPipeline.from_pretrained(model, torch_dtype='auto')

	return model.unet

# Has to be called individually for upblock, downblock, and midblock and embeddings needs to be hardcoded if we want to do layer-wise 
# Instead we hack each linear layer in the DM and perform quantization within the new classes forward methods
# This makes it so we don't have to define a pre-determined size for inps and outs (since that changes within blocks) and so we don't have to hardcode anything
@torch.no_grad()
def sdxl_sequential(model, dataloader, vae, tokenizer1, tokenizer2, dataset_class, max_token_length, resolution, debug_dataset, device, wbits: int, nsamples: int, true_sequential: bool, sym: bool, percdamp: float, groupsize: int, act_order: bool):
	# Prepare
	Args = namedtuple("args", "dataset_class max_token_length resolution debug_dataset")
	# args = Args(dataset_class, max_token_length, resolution, debug_dataset)
	# calibration_dataset = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])
	
	# dataloader = torch.utils.data.DataLoader(
    #     calibration_dataset,
    #     batch_size=1,
    #     shuffle=True,
    # )

	torch.cuda.empty_cache()

	quantizers = {}
	# Change iteration to running full model and keeping track of inps and outs and running quantization directly in the catcher 
	
	# Hijacks Linear Layers in unet, catches input and output and performs quantization
	# Done instead of iterating through layers and doing it one by one since DMs are not true sequential models
	# Requires full unet to be in memory; Mostly shouldn't be an issue since DMs can still fit on most consumer grade gpus
	class LinearQuantizer(nn.Module):
		def __init__(self, layer: nn.Linear, batch_size: int, idx: int, quantizers: dict, nsamples, wbits, sym, percdamp, groupsize, act_order):
			super().__init__()
			self.layer = layer
			self.quantizers = quantizers
			self.nsamples = nsamples
			self.wbits = wbits
			self.sym = sym
			self.percdamp = percdamp
			self.groupsize = groupsize
			self.act_order = act_order
			self.batch_size = batch_size
			self.idx = idx

		def forward(self, x):
			full = {name: m for name, m in self.layer.named_modules() if isinstance(m, nn.Linear)}

			sequential = [list(full.keys())]
			outs = []
			# For each subset of linear layers
			for names in sequential:
				subset = {n: full[n] for n in names}
				gptq = {}

				# Prepare a quantizer for each linear layer
				# Rather than doing it based off module name we do it based off idx since module name is not accessible from inside the linear layer
				for name in subset:
					gptq[f"{name}"] = GPTQ(subset[name])
					gptq[name].quantizer = Quantizer()
					gptq[name].quantizer.configure(self.wbits, perchannel=True, sym=self.sym, mse=False)
				
				def add_batch(name):
					def tmp(_, inp, out):
						gptq[name].add_batch(inp[0].data, out.data)
					return tmp

				handles = []
				for name in subset:
					handles.append(subset[name].register_forward_hook(add_batch(name)))
				for j in range(self.batch_size): #batch size
					outs.append(self.layer(x[j].unsqueeze(0))[0])  # TODO: Saving outs doesn't seem needed here?
				for h in handles:
					h.remove()
				# Once we add our batches of data to the quantizer we reset the outputs to return the correct dim outputs
				outs = []
				# With the data collected, quantize the layers
				for i, name in enumerate(subset):
					print(i, name)
					scale, zero = gptq[name].fasterquant(percdamp=self.percdamp, groupsize=self.groupsize, actorder=self.act_order)
					self.quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer, scale, zero)
					gptq[name].free()
			outs = []
			# Save outputs of the layer after quantization, so we can feed them into the next layer
			for j in range(self.batch_size): #batch size
				outs.append(self.layer(x[j].unsqueeze(0))[0])
			# free up memory
			del gptq 
			torch.cuda.empty_cache()
			# return output as you would in a normal linear layer
			return torch.cat(outs)
	
	# replace all instances of nn.Linear with our custom class
	for name, m in model.named_modules():
		if not isinstance(m, nn.Linear):
			continue
		
		# Replace the linear layer with a quantized one
		newlayer = LinearQuantizer(m, 1, quantizers, nsamples, wbits, sym, 
							 		percdamp, groupsize, act_order)
		parent_name = name.rsplit('.', 1)[0]
		parent = model.get_submodule(parent_name)

		#print(f"Replacing {name} with quant; parent: {parent_name}, child's name: {name[len(parent_name) + 1:]}")

		setattr(parent, name[len(parent_name) + 1:], newlayer)

	# once LinearQuantizer is put in place we run the model on the dataset to collect quantizers data
	# region sdxl dataset pass
	# for batch in dataloader:
	# 	try:
	# 		with torch.no_grad():
	# 			if "latents" in batch and batch["latents"] is not None:
	# 				latents = batch["latents"].to(model.device).to(dtype=model.dtype)
	# 			else:
	# 				with torch.no_grad():
	# 					# latent encoding
	# 					latents = vae.encode(batch["images"].to(vae.dtype)).latent_dist.sample().to(model.dtype)

	# 					# NaN check
	# 					if torch.any(torch.isnan(latents)):
	# 						latents = torch.nan_to_num(latents, 0, out=latents)
	# 			# Latent scale factor
	# 			latents = latents * 0.13025

	# 			if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
	# 				input_ids1 = batch["input_ids"]
	# 				input_ids2 = batch["input_ids2"]
	# 				with torch.set_grad_enabled(False):
	# 					input_ids1 = input_ids1.to(model.device)
	# 					input_ids2 = input_ids2.to(model.device)
	# 					# unwrap_model is fine for models not wrapped by accelerator
	# 					encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
	# 						max_token_length,
	# 						input_ids1,
	# 						input_ids2,
	# 						tokenizer1,
	# 						tokenizer2,
	# 						tokenizer1,
	# 						tokenizer2,
	# 						model.dtype,
	# 						accelerator=None,
	# 					)
	# 			else:
	# 				encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(model.device).to(model.dtype)
	# 				encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(model.device).to(model.dtype)
	# 				pool2 = batch["text_encoder_pool2_list"].to(model.device).to(model.dtype)

	# 			# get size embeddings
	# 			orig_size = batch["original_sizes_hw"]
	# 			crop_size = batch["crop_top_lefts"]
	# 			target_size = batch["target_sizes_hw"]
	# 			embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, model.device).to(model.dtype)

	# 			# concat embeddings
	# 			vector_embedding = torch.cat([pool2, embs], dim=1).to(model.dtype)
	# 			text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(model.dtype)

	# 			# Sample noise, sample a random timestep for each image, and add noise to the latents,
	# 			# with noise offset and/or multires noise if specified
	# 			#TODO: Args needs to be updated
	# 			noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

	# 			noisy_latents = noisy_latents.to(model.dtype)

	# 			noise_pred = model(noisy_latents, timesteps, text_embedding, vector_embedding)
	# 	except ValueError:
	# 		pass
	# endregion
	
	for batch in dataloader:
		try:
			model(batch.to(device))
		except ValueError:
			pass
	# Once we finish running over our dataset we replace our custom linear layers back with the originals
	return quantizers


def sdxl_pack(model, quantizers, wbits: int, groupsize: int):
	# Find all the quantized layers
	layers = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
	layers = {n: layers[n] for n in quantizers}

	# Replace all applicable instances of Linear with QuantLinear in the model
	quant_linear.make_quant(model, wbits, groupsize)

	for name, m in tqdm(model.named_modules(), total=len(list(model.named_modules()))):
		if not isinstance(m, QuantLinear):
			continue

		quantizer, scale, zero = quantizers[name]
		quantizer, scale, zero = quantizer.cpu(), scale.cpu(), zero.cpu()
		pack_linear(m, layers[name].weight.data, scale, zero, m.bias)


def pack_linear(quant, weights: torch.FloatTensor, scales: torch.FloatTensor, zeros, bias: Optional[torch.FloatTensor]):
	"""
	Packs the quantized weights, scales, and zero points into a QuantLinear layer
	"""
	scales = scales.t().contiguous()
	zeros = zeros.t().contiguous()
	scale_zeros = zeros * scales

	quant.scales = scales.clone().to(torch.float16)

	if quant.bias is not None:
		quant.bias = bias.clone().to(torch.float16)
	
	# Round weights to nearest integer based on scale and zero point
	# Each weight will be one int, but should not exceed quant.bits
	intweight = []
	for idx in range(quant.infeatures):
		g_idx = idx // quant.groupsize
		# TODO: This is oddly complex.  The `gptq.quantize` function does `return scale * (q - zero)`, so shouldn't
		# this just be `q = torch.round((weights[:,idx] / scales[g_idx]) + zero[g_idx])`?
		q = torch.round((weights[:,idx] + scale_zeros[g_idx]) / scales[g_idx]).to(torch.int32)
		intweight.append(q[:,None])
	intweight = torch.cat(intweight,dim=1)
	intweight = intweight.t().contiguous()

	# Now pack the weights into uint32's
	#qweight = torch.zeros((intweight.shape[0] // 32 * quant.bits, intweight.shape[1]), dtype=torch.int32)
	quant.qweight.zero_()
	i = 0
	row = 0
	while row < quant.qweight.shape[0]:
		if quant.bits in [2,4,8]:
			for j in range(i, i + (32 // quant.bits)):
				quant.qweight[row] |= intweight[j] << (quant.bits * (j - i))
			i += 32 // quant.bits
			row += 1
		else:
			raise NotImplementedError("Only 2,4,8 bits are supported.")
	
	# Subtract 1 from the zero point
	zeros = zeros - 1

	# Pack the zero points into uint32's
	zeros = zeros.to(torch.int32)
	#qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 256 * (self.bits * 8)), dtype=np.uint32)
	quant.qzeros.zero_()
	i = 0
	col = 0
	while col < quant.qzeros.shape[1]:
		if quant.bits in [2,4,8]:
			for j in range(i, i + (32 // quant.bits)):
				quant.qzeros[:, col] |= zeros[:, j] << (quant.bits * (j - i))
			i += 32 // quant.bits
			col += 1
		else:
			raise NotImplementedError("Only 2,4,8 bits are supported.")

if __name__ == '__main__':
	main()