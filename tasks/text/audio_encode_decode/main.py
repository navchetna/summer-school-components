from audiocodec import AudioCodec

NQ = 8
CODEBOOK_OFFSET = True
model_name = "encodec_24khz"
audio_codec = AudioCodec(model_name)

f_in_enc = "test_wavs/61_70970_000007_000001.wav"
f_out_enc = "encoded.json"
audio_codec.encode_file(f_in_enc, f_out_enc, n_q=NQ, codebook_offset=CODEBOOK_OFFSET)

f_out_dec = "decoded.wav"
audio_codec.decode_file(f_out_enc, f_out_dec, codebook_offset=CODEBOOK_OFFSET)
print("Encoding and decoding completed successfully!")
