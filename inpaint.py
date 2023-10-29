import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import numpy as np
from PIL import Image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("output/image_sparse_8/test/ours_30000/renders/00000.png").convert("RGB")
mask_image = load_image("output/image_sparse_8/test/ours_30000/mask/00000.png").convert("RGB")
# flip mask
mask_image = np.array(mask_image)
mask_image = ((mask_image < 0.95*255) * 255).astype(np.uint8)
mask_image = Image.fromarray(mask_image)

generator = torch.Generator("cuda").manual_seed(92)
prompt = "a bicycle, park, outdoor scene, high quality"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator, 
                 height=1064, width=1600).images[0]

# save image
image.save('temp.png')
