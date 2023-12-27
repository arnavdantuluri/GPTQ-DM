import torch
from unet_pt import UNet2DConditionModel
from collections import namedtuple
from PIL import Image, ImageChops
import torch.nn as nn
# Load weights from the original model
from diffusers import DiffusionPipeline
import sys
from datetime import datetime
from sfast.utils.term_image import print_image
from sfast.jit.trace_helper import lazy_trace

sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

fuse = False
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(42)

unet_new = UNet2DConditionModel().half().cuda()
unet_new.load_state_dict(pipe.unet.state_dict())

prompt = "a photo of an astronaut riding a horse on mars"
# setattr(pipe, "_execution_device", "cuda")
# image = pipe(prompt, generator=generator).images[0]
# print_image(image, max_width=100)
# del image 
# del pipe
sample = torch.rand(2, 4, 128, 128).cuda().half()
timesteps = torch.rand([]).cuda()
encoder_hidden_states = torch.rand(2, 77, 2048).cuda().half()
added_cond_kwargs = {
    'text_embeds': torch.rand(2, 1280).cuda().half(),
    'time_ids': torch.rand(2, 6).cuda().half(),
}
# With the traced function we can overwrite regular linear with quantized linear
traced = lazy_trace(unet_new.forward)

l = [name for _, name in pipe.unet.named_modules() if isinstance(name, nn.Linear) ]
print(l)