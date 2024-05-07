#!usr/bin/env python3
"""
 Copyright (c) 2019-2023 Intel Corporation

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

import logging as log
import sys
from time import perf_counter
from argparse import ArgumentParser
from pathlib import Path
import os

import cv2

from instance_segmentation_demo.tracker import StaticIOUTracker

OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"

sys.path.append(f'{OPEN_MODEL_ZOO_DIR}/demos/common/python')
sys.path.append(f'{OPEN_MODEL_ZOO_DIR}/demos/common/python/openvino/model_zoo')


from model_api.models import MaskRCNNModel, YolactModel, OutputTransform
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.performance_metrics import PerformanceMetrics


import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import InstanceSegmentationVisualizer

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

LOOP = False
DEVICE = "CPU"
NUM_STREAMS  = ''
NUM_THREADS  = None
MODEL_DIR = f"{OPEN_MODEL_ZOO_DIR}/demos/instance_segmentation_demo/python"
MODEL = f'{MODEL_DIR}/models/instance-segmentation-security-0002.xml'
LAYOUT = None
INFER_REQUESTS = 0
PROB_THRESHOLD = 0.5
SHOW_BOXES = False
SHOW_SCORES = False
OUTPUT_RESOLUTION = None
LABELS = f'{OPEN_MODEL_ZOO_DIR}/data/dataset_classes/coco_80cl_bkgr.txt'
ADAPTER = 'openvino'
OUTPUT = None
OUTPUT_LIMIT = 1000
NO_SHOW = True
NO_TRACK =  False
UTILIZATION_MONITORS = None
RAW_OUTPUT_MESSAGE = False




def get_model(model_adapter, configuration):
    inputs = model_adapter.get_input_layers()
    outputs = model_adapter.get_output_layers()
    if len(inputs) == 1 and len(outputs) == 4 and 'proto' in outputs.keys():
        return YolactModel(model_adapter, configuration)
    return MaskRCNNModel(model_adapter, configuration)


def print_raw_results(boxes, classes, scores, frame_id):
    log.debug('  -------------------------- Frame # {} --------------------------  '.format(frame_id))
    log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
    for box, cls, score in zip(boxes, classes, scores):
        log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

#Namspace(adapter='openvino', input='/home/alok/sample.jpg', device='CPU', labels='/home/alok/open_model_zoo/data/dataset_classes/coco_80cl_bkgr.txt', prob_threshold=0.5, no_track=False, show_scores=False, show_boxes=False, layout=None, num_infer_requests=0, num_streams='', num_threads=None, loop=False, output=None, output_limit=1000, no_show=False, output_resolution=None, utilization_monitors=None, raw_output_message=False)e


def init_model(DEVICE, NUM_THREADS,NUM_STREAMS):
    """ Initialises the model and the pipeline """
    if ADAPTER == 'openvino':
        plugin_config = get_user_config(DEVICE, NUM_STREAMS, NUM_THREADS)
        model_adapter = OpenvinoAdapter(create_core(), MODEL, device=DEVICE, plugin_config=plugin_config,
                                        max_num_requests= INFER_REQUESTS,
                                        model_parameters={'input_layouts': LAYOUT})
    elif ADAPTER == 'ovms':
        model_adapter = OVMSAdapter(MODEL)

    configuration = {
        'confidence_threshold': PROB_THRESHOLD,
        'path_to_labels': LABELS,
    }
    model = get_model(model_adapter, configuration)
    model.log_layers_info()

    return model


def get_metric_dict(cap,pipeline,render_metrics):
 
    metric_dict = {
                    "reader_latency": cap.reader_metrics.get_latency(),
                    "preprocess_metrics": pipeline.preprocess_metrics.get_latency(),
                    "inference_metrics": pipeline.inference_metrics.get_latency(),
                    "postprocess_metrics": pipeline.postprocess_metrics.get_latency(),
                    "render_metrics": render_metrics.get_latency(),

                  }
    return metric_dict

class instance_segmentation:
    def __init__(self, DEVICE:str, NUM_THREADS: int, NUM_STREAMS: str):
        self.device = DEVICE
        self.num_threads = NUM_THREADS
        self.num_streams = NUM_STREAMS
        self.model = init_model(DEVICE,NUM_THREADS,NUM_STREAMS)

    def get_segments(self,file_path):
        cap = open_images_capture(file_path, LOOP) #args.input is just path of a  file ( mp4,  jpeg) 
        pipeline = AsyncPipeline(self.model)
        next_frame_id = 0
        next_frame_id_to_show = 0

        tracker = None
        if not NO_TRACK and cap.get_type() in {'VIDEO', 'CAMERA'}:
            tracker = StaticIOUTracker()
        visualizer = InstanceSegmentationVisualizer(self.model.labels, SHOW_BOXES, SHOW_SCORES)

        metrics = PerformanceMetrics()
        render_metrics = PerformanceMetrics()
        presenter = None
        output_transform = None
        video_writer = cv2.VideoWriter()

        while True:
            if pipeline.is_ready():
                # Get new image/frame
                start_time = perf_counter()
                frame = cap.read()
                if frame is None:
                    if next_frame_id == 0:
                        raise ValueError("Can't read an image from the input")
                    break
                if next_frame_id == 0:
                    output_transform = OutputTransform(frame.shape[:2], OUTPUT_RESOLUTION)
                    if OUTPUT_RESOLUTION:
                        output_resolution = output_transform.new_resolution
                    else:
                        output_resolution = (frame.shape[1], frame.shape[0])
                    presenter = monitors.Presenter(UTILIZATION_MONITORS, 55,
                                                   (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                    if OUTPUT and not video_writer.open(OUTPUT, cv2.VideoWriter_fourcc(*'MJPG'),
                                                             cap.fps(), tuple(output_resolution)):
                        raise RuntimeError("Can't open video writer")
                # Submit for inference
                pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
                next_frame_id += 1
            else:
                # Wait for empty request
                pipeline.await_any()

            if pipeline.callback_exceptions:
                raise pipeline.callback_exceptions[0]
            # Process all completed requests
            results = pipeline.get_result(next_frame_id_to_show)
            if results:
                (scores, classes, boxes, masks), frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']

                if RAW_OUTPUT_MESSAGE:
                    print_raw_results(boxes, classes, scores, next_frame_id_to_show)

                rendering_start_time = perf_counter()
                masks_tracks_ids = tracker(masks, classes) if tracker else None
                frame = visualizer(frame, boxes, classes, scores, masks, masks_tracks_ids)
                render_metrics.update(rendering_start_time)

                res =cv2.imwrite("output.jpeg", frame)
                print("Image save: " + str(res))
                presenter.drawGraphs(frame)
                metrics.update(start_time, frame)

                if video_writer.isOpened() and (OUTPUT_LIMIT <= 0 or next_frame_id_to_show <= OUTPUT_LIMIT - 1):
                    video_writer.write(frame)
                next_frame_id_to_show += 1

                if not NO_SHOW:
                    cv2.imshow('Instance Segmentation results', frame)
                    key = cv2.waitKey(1)
                    if key == 27 or key == 'q' or key == 'Q':
                        break
                    presenter.handleKey(key)

                scores = results[0][0].tolist()
                classes = results[0][1].tolist()
                boxes = results[0][2].tolist()
                masks  = results[0][3]
                for i in range(len(masks)):
                    masks[i] = masks[i].tolist()

                metrics.log_total()
                metric_dict = get_metric_dict(cap,pipeline,render_metrics)
                return (scores, classes, boxes, masks,metric_dict)



        pipeline.await_all()
        if pipeline.callback_exceptions:
            raise pipeline.callback_exceptions[0]
        # Process completed requests
        for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
            results = pipeline.get_result(next_frame_id_to_show)
            (scores, classes, boxes, masks), frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if RAW_OUTPUT_MESSAGE:
                print_raw_results(boxes, classes, scores, next_frame_id_to_show)

            rendering_start_time = perf_counter()
            masks_tracks_ids = tracker(masks, classes) if tracker else None
            frame = visualizer(frame, boxes, classes, scores, masks, masks_tracks_ids)
            render_metrics.update(rendering_start_time)

            res =cv2.imwrite("output.jpeg", frame) #TODO: remove this later
            print("Image save: " + str(res)) #TODO: remove this later
            presenter.drawGraphs(frame)
            metrics.update(start_time, frame)

            if video_writer.isOpened() and (OUTPUT_LIMIT <= 0 or next_frame_id_to_show <= OUTPUT_LIMIT - 1):
                video_writer.write(frame)

            if not NO_SHOW:
                cv2.imshow('Instance Segmentation results', frame)
                cv2.waitKey(1)

            scores = results[0][0].tolist()
            classes = results[0][1].tolist()
            boxes = results[0][2].tolist()
            masks  = results[0][3]
            for i in range(len(masks)):
                masks[i] = masks[i].tolist()

            metrics.log_total()
            metric_dict = get_metric_dict(cap,pipeline,render_metrics)
            return (scores, classes, boxes, masks,metric_dict)



if __name__ == '__main__':

    ic_obj = instance_segmentation(DEVICE = DEVICE, NUM_THREADS = NUM_THREADS, NUM_STREAMS = NUM_STREAMS)
    res = ic_obj.get_segments(sys.argv[1])
    print(res)

