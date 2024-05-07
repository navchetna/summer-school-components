#!/usr/bin/env python3

"""
 Copyright (c) 2020-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
import os
import logging as log
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

from tqdm import tqdm
import numpy as np
import wave
from openvino.runtime import Core, get_version

from models.forward_tacotron_ie import ForwardTacotronIE
from models.mel2wave_ie import WaveRNNIE, MelGANIE
from utils.gui import init_parameters_interactive

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def save_wav(x):
    sr = 22050

    with wave.open("/tmp/out.wav", "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(x.tobytes())




class text_to_speech:


    def __init__(self,DEVICE = "CPU",ALPHA = 1.0):

        self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
        self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
        self.MODEL_DURATION = f"{self.OPEN_MODEL_ZOO_DIR}/demos/text_to_speech_demo/python/models/text-to-speech-en-0001-duration-prediction.xml"
        self.MODEL_FORWARD = f"{self.OPEN_MODEL_ZOO_DIR}/demos/text_to_speech_demo/python/models/text-to-speech-en-0001-regression.xml"
        self.MODEL_MELGAN = f"{self.OPEN_MODEL_ZOO_DIR}/demos/text_to_speech_demo/python/models/text-to-speech-en-0001-generation.xml"
        self.OUT =  "out.wav"
        self.DEVICE =  DEVICE
        self.MODEL_UPSAMPLE =  None
        self.MODEL_RNN =  None
        self.UPSAMPLER_WIDTH =  1
        self.SPEAKER_ID =  19
        self.ALPHA = ALPHA


    def _is_correct(self):
        if not ((self.self.MODEL_MELGAN is None and self.MODEL_RNN is not None and MODEL_UPSAMPLE is not None) or
                (self.MODEL_MELGAN is not None and MODEL_RNN is None and MODEL_UPSAMPLE is None)):
            log.error('Can not use m_rnn and m_upsample with m_melgan. Define m_melgan or [m_rnn, m_upsample]')
            return False
        if self.ALPHA < 0.5 or self.ALPHA > 2.0:
            log.error('Can not use time coefficient less than 0.5 or greater than 2.0')
            return False
        if self.SPEAKER_ID < -1 or self.SPEAKER_ID > 39:
            log.error('Mistake in the range of speaker_id. Speaker_id should be -1 (GUI regime) or in range [0,39]')
            return False

        return True

    def _parse_input(self,text:str):

        text_sentences = text.split("\n")
        if(text_sentences[-1] == ''):
            text_sentences = text_sentences[0: len(text_sentences)-1]

        return text_sentences


    def get_speech(self,INPUT_TEXT):

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        if self.MODEL_MELGAN is not None:
            vocoder = MelGANIE(self.MODEL_MELGAN, core, device = self.DEVICE)
        else:
            vocoder = WaveRNNIE(self.MODEL_UPSAMPLE, self.MODEL_RNN, core, device=self.DEVICE,
                                upsampler_width= self.UPSAMPLER_WIDTH)

        forward_tacotron = ForwardTacotronIE(self.MODEL_DURATION, self.MODEL_FORWARD, core, self.DEVICE, verbose=False)

        audio_res = np.array([], dtype=np.int16)

        speaker_emb = None
        if forward_tacotron.is_multi_speaker:
            #if SPEAKER_ID == -1:
                #interactive_parameter = init_parameters_interactive(args)
                #ALPHA= 1.0 / interactive_parameter["speed"]
                #speaker_emb = forward_tacotron.get_pca_speaker_embedding(interactive_parameter["gender"],
                                                                         #interactive_parameter["style"])
            #else:
                speaker_emb = [forward_tacotron.get_speaker_embeddings()[self.SPEAKER_ID, :]]

        len_th = 80

        input_data = self._parse_input(INPUT_TEXT)

        time_forward = 0
        time_wavernn = 0

        time_s_all = perf_counter()
        count = 0
        for line in input_data:
            count += 1
            line = line.rstrip()
            log.info("Process line {0} with length {1}.".format(count, len(line)))

            if len(line) > len_th:
                texts = []
                prev_begin = 0
                delimiters = '.!?;:,'
                for i, c in enumerate(line):
                    if (c in delimiters and i - prev_begin > len_th) or i == len(line) - 1:
                        texts.append(line[prev_begin:i + 1])
                        prev_begin = i + 1
            else:
                texts = [line]

            for text in tqdm(texts):
                time_s = perf_counter()
                mel = forward_tacotron.forward(text, alpha=self.ALPHA, speaker_emb=speaker_emb)
                time_forward += perf_counter() - time_s

                time_s = perf_counter()
                audio = vocoder.forward(mel)
                time_wavernn += perf_counter() - time_s

                audio_res = np.append(audio_res, audio)

        total_latency = (perf_counter() - time_s_all) * 1e3
        log.info("Metrics report:")
        log.info("\tLatency: {:.1f} ms".format(total_latency))
        log.debug("\tVocoder time: {:.1f} ms".format(time_wavernn * 1e3))
        log.debug("\tForwardTacotronTime: {:.1f} ms".format(time_forward * 1e3))

        metrics= {"Latency": total_latency, "Vocoder time": time_wavernn*1e3, "ForwardTacotronTime": time_forward*1e3}

        save_wav(audio_res)
        wav_file = open("/tmp/out.wav","rb")
        wav_file_bytes = wav_file.read()
        wav_file.close()
        return (wav_file_bytes, metrics)
        


if __name__ == '__main__':

    tsp_obj = text_to_speech()
    res = tsp_obj.get_speech(sys.argv[1])
