#!/usr/bin/env python3

import sys, os
from time import perf_counter
import logging as log
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import cv2
import numpy as np

from openvino.runtime import Core, get_version
from utils.codec import CTCCodec

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.DEBUG,
                stream=sys.stdout)

OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
MODEL_DIR = f"{OPEN_MODEL_ZOO_DIR}/demos/handwritten_text_recognition_demo/python"

MODEL_PATH = f"{MODEL_DIR}/models/handwritten-english-recognition-0001.xml"  # Path to an .xml file with a trained model (Required)
INPUT_IMAGE_PATH = f"{MODEL_DIR}/data/handwritten_english_test.png"  # Path to an image to infer (Required)
DEVICE = "CPU"  # Specify the target device to infer on; CPU, GPU, or HETERO is acceptable. The demo will look for a suitable plugin for the specified device. Default value is CPU (Optional)
NUM_ITERATIONS = 1  # Number of inference iterations (Optional)
CHARLIST_PATH = f"{OPEN_MODEL_ZOO_DIR}/data/dataset_classes/gnhk.txt"
DESIGNATED_CHARACTERS_PATH = None  # Path to the designated character file (Optional)
TOP_K = 20  # Top k steps in looking up the decoded character until a designated one is found (Optional)
OUTPUT_BLOB_NAME = None  # Name of the output layer of the model. Default is None, in which case the demo will read the output name from the model, assuming there is only 1 output layer (Optional)


def get_characters(charlist):
    '''Get characters'''
    with open(charlist, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)


def preprocess_input(image_name, height, width):
    src = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise RuntimeError(f"Failed to imread {image_name}")
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)
    rsz = cv2.resize(src, (tw, height),
                     interpolation=cv2.INTER_AREA).astype(np.float32)
    # [h,w] -> [c,h,w]
    img = rsz[None, :, :]
    _, h, w = img.shape
    # right edge padding
    pad_img = np.pad(img, ((0, 0), (0, height - h), (0, width - w)),
                     mode='edge')
    return pad_img


def transcribe(input_image=INPUT_IMAGE_PATH, device=DEVICE, top_k=TOP_K, num_iterations=NUM_ITERATIONS):
    # Plugin initialization
    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    if 'GPU' in device:
        core.set_property("GPU", {
            "GPU_ENABLE_LOOP_UNROLLING": "NO",
            "CACHE_DIR": "./"
        })

    # Read IR
    log.info('Reading model {}'.format(MODEL_PATH))
    model = core.read_model(MODEL_PATH)

    if len(model.inputs) != 1:
        raise RuntimeError("Demo supports only single input topologies")
    input_tensor_name = model.inputs[0].get_any_name()

    if OUTPUT_BLOB_NAME is not None:
        output_tensor_name = OUTPUT_BLOB_NAME
    else:
        if len(model.outputs) != 1:
            raise RuntimeError("Demo supports only single output topologies")
        output_tensor_name = model.outputs[0].get_any_name()

    characters = get_characters(CHARLIST_PATH)
    codec = CTCCodec(characters, DESIGNATED_CHARACTERS_PATH, top_k)
    if len(codec.characters) != model.output(output_tensor_name).shape[2]:
        raise RuntimeError(
            "The text recognition model does not correspond to decoding character list"
        )

    input_batch_size, input_channel, input_height, input_width = model.inputs[
        0].shape

    # Read and pre-process input image (NOTE: one image only)
    preprocessing_start_time = perf_counter()
    input_image = preprocess_input(input_image,
                                   height=input_height,
                                   width=input_width)[None, :, :, :]
    preprocessing_total_time = perf_counter() - preprocessing_start_time
    if input_batch_size != input_image.shape[0]:
        raise RuntimeError(
            "The model's input batch size should equal the input image's batch size"
        )
    if input_channel != input_image.shape[1]:
        raise RuntimeError(
            "The model's input channel should equal the input image's channel")

    # Loading model to the plugin
    compiled_model = core.compile_model(model, device)
    infer_request = compiled_model.create_infer_request()
    log.info('The model {} is loaded to {}'.format(MODEL_PATH, device))

    results = []
    # Start sync inference
    start_time = perf_counter()
    for _ in range(num_iterations):
        infer_request.infer(inputs={input_tensor_name: input_image})
        preds = infer_request.get_tensor(output_tensor_name).data[:]
        result = codec.decode(preds)
        results.append(result)
    total_latency = ((perf_counter() - start_time) / num_iterations +
                     preprocessing_total_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))

    return results


if __name__ == '__main__':
    print(transcribe())
