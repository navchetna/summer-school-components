import logging as log
import sys
import os
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / 'common/python'))
sys.path.append(
    str(
        Path(__file__).resolve().parents[1] /
        'common/python/openvino/model_zoo'))

from model_api.models import ImageModel
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter

from images_capture import open_images_capture

from helpers import log_latency_per_stage

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.DEBUG,
                stream=sys.stdout)

ARCHITECTURES = {
    'ae': 'HPE-assosiative-embedding',
    'higherhrnet': 'HPE-assosiative-embedding',
    'openpose': 'openpose'
}

OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
MODEL_DIR = f"{OPEN_MODEL_ZOO_DIR}/demos/human_pose_estimation_demo/python"

# Common model options
PROB_THRESHOLD = 0.1
TARGET_SIZE = None
INPUT_LAYOUT = None

# Inference options
NUM_INFER_REQUESTS = 0
NUM_STREAMS = ""
NUM_THREADS = None

# Debug options
RAW_OUTPUT_MESSAGE = True

# Path to an .xml file with a trained model (Required)
MODEL_PATH = f"{MODEL_DIR}/models/human-pose-estimation-0005.xml"

# Specify model architecture type (Required) (Choices: ae, higherhrnet, openpose)
ARCHITECTURE_TYPE = "ae"

# An input to process (Required)
INPUT = f"{MODEL_DIR}/people-detection.mp4"

# Enable reading the input in a loop (Optional)
LOOP = False

# Specify the target device to infer on; CPU or GPU is acceptable. The demo will look for a suitable plugin for the device specified. Default value is CPU. (Optional)
DEVICE = "CPU"

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                    (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
          (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
          (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170,
                                                                    255))


def draw_poses(img,
               poses,
               point_score_threshold,
               output_transform,
               skeleton=default_skeleton,
               draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[
                    j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(
                        tuple(middle),
                        (int(length / 2), min(int(length / 50), stick_width)),
                        angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs,
                             tuple(points[i]),
                             tuple(points[j]),
                             color=colors[j],
                             thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def print_raw_results(poses, scores, frame_id):
    log.debug(
        ' ------------------- Frame # {} ------------------ '.format(frame_id))
    for pose, pose_score in zip(poses, scores):
        pose_str = ' '.join(
            '({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
        log.debug('{} | {:.2f}'.format(pose_str, pose_score))


def detect(input=INPUT,
           device=DEVICE,
           prob_threshold=PROB_THRESHOLD,
           target_size=TARGET_SIZE,
           input_layout=INPUT_LAYOUT,
           num_infer_requests=NUM_INFER_REQUESTS,
           num_streams=NUM_STREAMS,
           num_threads=NUM_THREADS,
           raw_output_message=RAW_OUTPUT_MESSAGE,
           architecture_type=ARCHITECTURE_TYPE):

    cap = open_images_capture(input, LOOP)
    next_frame_id = 1
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    plugin_config = get_user_config(device, num_streams, num_threads)
    model_adapter = OpenvinoAdapter(
        create_core(),
        MODEL_PATH,
        device=device,
        plugin_config=plugin_config,
        max_num_requests=num_infer_requests,
        model_parameters={'input_layouts': input_layout})

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    config = {
        'target_size': target_size,
        'aspect_ratio': frame.shape[1] / frame.shape[0],
        'confidence_threshold': prob_threshold,
        'padding_mode':
        'center' if architecture_type == 'higherhrnet' else None,
        'delta': 0.5 if architecture_type == 'higherhrnet' else None,
    }
    model = ImageModel.create_model(ARCHITECTURES[architecture_type],
                                    model_adapter, config)
    model.log_layers_info()

    hpe_pipeline = AsyncPipeline(model)
    hpe_pipeline.submit_data(frame, 0, {
        'frame': frame,
        'start_time': start_time
    })

    payload = []

    while True:
        if hpe_pipeline.callback_exceptions:
            raise hpe_pipeline.callback_exceptions[0]

        # Process all completed requests
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(poses):
                if raw_output_message: print_raw_results(poses, scores, next_frame_id_to_show)
                payload.append({
                    "poses": poses,
                    "scores": scores,
                    "frame_id": next_frame_id_to_show
                })

            rendering_start_time = perf_counter()
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)
            next_frame_id_to_show += 1

        if hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break

            # Submit for inference
            hpe_pipeline.submit_data(frame, next_frame_id, {
                'frame': frame,
                'start_time': start_time
            })
            next_frame_id += 1

        else:
            # Wait for an empty request
            hpe_pipeline.await_any()

    hpe_pipeline.await_all()
    if hpe_pipeline.callback_exceptions:
        raise hpe_pipeline.callback_exceptions[0]

    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        (poses, scores), frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(poses) != 0:
            payload.append({
                "poses": poses,
                "scores": scores,
                "frame_id": next_frame_id_to_show
            })

    metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          hpe_pipeline.preprocess_metrics.get_latency(),
                          hpe_pipeline.inference_metrics.get_latency(),
                          hpe_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())

    return payload

def main():
    print(detect())

if __name__ == '__main__':
    sys.exit(main() or 0)
