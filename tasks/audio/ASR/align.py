import whisperx
import intel_extension_for_pytorch as ipex
import torch
import requests


def load_model(config):
    model, metadata = whisperx.load_align_model(
        language_code=config["language"], device=config["device"]
    )
    return model, metadata


def optimize(model, config):
    print(f"Running IPEX optimization with compute_type {config['compute_type']}")
    ipex_feature_extractor = ipex.optimize(
        model.feature_extractor, dtype=config["compute_type"]
    )
    model.feature_extractor = ipex_feature_extractor
    print(f"Done IPEX optimization")
    return model


def amp_align(audio_file, transcription, model, metadata, config):
    with torch.no_grad(), torch.cpu.amp.autocast():
        transcript_aligned = whisperx.align(
            transcription["segments"], model, metadata, audio_file, config["device"]
        )
    return transcript_aligned


def align(audio_file, transcription, model, metadata, config):
    transcript_aligned = whisperx.align(
        transcription["segments"], model, metadata, audio_file, config["device"]
    )
    return transcript_aligned


def align_transcription(audio_file, transcription, config, model, metadata):
    if config["optimize_model"]:
        return amp_align(audio_file, transcription, model, metadata, config)
    else:
        return align(audio_file, transcription, model, metadata, config)


def run_model(config, audio_file):
    model, metadata = load_model(config)
    if config["optimize_model"]:
        model = optimize(model, config)
    transcription = requests.get("http://localhost:8000/transcribe").json()
    output = align_transcription(audio_file, transcription, config, model, metadata)
    return output
