import whisperx


def run_model(config, audio_file):
    if config["pipeline"] == "whisperx":
        model = whisperx.load_model(
            config["model"], config["device"], compute_type=config["compute_type"]
        )
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=config["batch_size"])
        return result


