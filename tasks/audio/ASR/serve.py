from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import ray
from ray import serve

import time
import torch

# Model files
from transcript import run_model
from align import run_model as align_run

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@serve.deployment
@serve.ingress(app)
class ServeWhisper:
    @app.get("/transcribe")
    def run_transcribe(self):
        config_transcription = {
            "device": "cpu",
            "model": "tiny.en",
            "compute_type": "float32",
            "pipeline": "whisperx",
            "batch_size": 16,
            "optimize_model": True,
        }
        sample_audio_file = "./samples/sample.wav"
        result = run_model(config_transcription, sample_audio_file)
        return result

    @app.get("/align")
    def run_align(self):
        config_alignment = {
            "device": "cpu",
            "compute_type": torch.bfloat16,
            "language": "en",
            "optimize_model": True,
        }
        sample_audio_file = "./samples/sample.wav"
        result = align_run(config_alignment, sample_audio_file)
        return result


ray.init()
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": "8000"})
serve.run(ServeWhisper.bind(), name="whisper", route_prefix="/")

count = 1
while True:
    count += 1
    if count % 1000 == 0:
        time.sleep(1)
