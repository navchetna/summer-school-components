from diffusers import StableDiffusionPipeline
import torch

class Text2Image:
    def __init__(self):
        self.model = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model, torch_dtype=torch.float32, cache_dir="my_models/")
        
    def generate(self, prompt = ""):
        return self.pipe(prompt).images[0]

    def save_image(image):    
        image.save("generated_image.png")
