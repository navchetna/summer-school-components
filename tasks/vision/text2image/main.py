from optimum.intel import OVStableDiffusionXLPipeline
import numpy as np

model_id = "FFusion/FFusionXL-BASE"
prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a beautiful cyber female wearing a black corset and pink latex shirt, scifi best quality, intricate details."
np.random.seed(0)

base = OVStableDiffusionXLPipeline.from_pretrained(model_id, export=False, compile=False)
base.compile()
image1 = base(prompt, num_inference_steps=50).images[0]
image1.save("sdxl_without_textual_inversion.png")
