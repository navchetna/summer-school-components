#!/usr/bin/env python3
"""
 Copyright (c) 2018-2023 Intel Corporation

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
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import cv2
import json
import numpy as np
from openvino.runtime import Core, get_version
import logging as log
import os
import os.path as osp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
sys.path.append(str(Path(__file__).resolve().parents[2] / '/home/develop/open_model_zoo/demos/common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / '/home/develop/open_model_zoo/demos/common/python/openvino/model_zoo'))
#sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
#sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))
import monitors
from helpers import resolution
from images_capture import open_images_capture
from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.models.utils import resize_image

import logging as log
from openvino.runtime import AsyncInferQueue

from openvino.runtime import PartialShape

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']





class Module:
    def __init__(self, core, model_path, model_type):
        self.core = core
        self.model_type = model_type
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        self.model_path = model_path
        self.active_requests = 0
        self.clear()

    def deploy(self, device, max_requests=1):
        self.max_requests = max_requests
        compiled_model = self.core.compile_model(self.model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_queue = AsyncInferQueue(compiled_model, self.max_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(self.model_type, self.model_path, device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.results[self.output_tensor]

    def enqueue(self, input):
        if self.max_requests <= self.active_requests:
            log.warning('Processing request rejected - too many requests')
            return False

        self.infer_queue.start_async(input, self.active_requests)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return
        self.infer_queue.wait_all()
        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return [v for _, v in sorted(self.outputs.items())]

    def clear(self):
        self.outputs = {}

    def infer(self, inputs):
        self.clear()
        self.start_async(*inputs)
        return self.postprocess()

class FaceDetector(Module):
    class Result:
        OUTPUT_SIZE = 7

        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4])) # (x, y)
            self.size = np.array((output[5], output[6])) # (w, h)

        def rescale_roi(self, roi_scale_factor=1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)

    def __init__(self, core, model, input_size, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(core, model, 'Face Detection')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape({self.input_tensor_name: PartialShape([1, 3, *input_size])})
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError("Both input height and width should be positive for Face Detector reshape")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape
        if len(self.output_shape) != 4 or self.output_shape[3] != self.Result.OUTPUT_SIZE:
            raise RuntimeError("The model expects output shape with {} outputs".format(self.Result.OUTPUT_SIZE))

        if confidence_threshold > 1.0 or confidence_threshold < 0:
            raise ValueError("Confidence threshold is expected to be in range [0; 1]")
        if roi_scale_factor < 0.0:
            raise ValueError("Expected positive ROI scale factor")

        self.confidence_threshold = confidence_threshold
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        self.input_size = frame.shape
        return Face_Detector.resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(FaceDetector, self).enqueue({self.input_tensor_name: input})

    def postprocess(self):
        outputs = self.get_outputs()[0]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break # results are sorted by confidence decrease

            result.resize_roi(self.input_size[1], self.input_size[0])
            result.rescale_roi(self.roi_scale_factor)
            result.clip(self.input_size[1], self.input_size[0])
            results.append(result)

        return results


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self,m_fd,fd_input_size,t_fd,exp_r_fd,d_fd,allow_grow,no_show):
        self.allow_grow = allow_grow and not no_show

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core,m_fd,
                                          fd_input_size,
                                          confidence_threshold=t_fd,
                                          roi_scale_factor=exp_r_fd)
    
        self.face_detector.deploy(d_fd)
        
    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        return rois

class Face_Detector:

    def __init__(self):
            self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
            self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
            self.model=None
            self.IN=None
            #self.DEVICE = DEVICE
            self.list_models_supported="models.txt"
            self.at="ssd"
            self.labels=None
            self.adapter="openvino"
            self.prob_threshold=0.5
            self.anchors=None
            self.masks=None
            self.num_classes=None
            self.num_streams="4"
            self.num_infer_requests=0
            self.num_threads=None
            self.layout=None
            self.resize_type=None
            self.mean_values=None
            self.scale_values=None
            self.reverse_input_channels=False
            self.input_size=(300, 300)
            self.raw_output_message=True
            self.output_limit=1000
            self.no_show=True
            self.output_resolution=None
            self.utilization_monitors=None
            self.output="output.mp4"
            self.loop=False
            self.allow_grow=False
            self.t_fd=0.6
            self.exp_r_fd=1.15
            self.fd_input_size=(0,0)
            self.d_fd='CPU'
            self.crop_size=(0,0)

    def crop(frame, roi):
        p1 = roi.position.astype(int)
        p1 = np.clip(p1, [0, 0], [frame.shape[1], frame.shape[0]])
        p2 = (roi.position + roi.size).astype(int)
        p2 = np.clip(p2, [0, 0], [frame.shape[1], frame.shape[0]])
        return frame[p1[1]:p2[1], p1[0]:p2[0]]


    def cut_rois(frame, rois):
        return [crop(frame, roi) for roi in rois]


    def resize_input(image, target_shape, nchw_layout):
        if nchw_layout:
            _, _, h, w = target_shape
        else:
            _, h, w, _ = target_shape
        resized_image = resize_image(image, (w, h))
        if nchw_layout:
            resized_image = resized_image.transpose((2, 0, 1)) # HWC->CHW
        resized_image = resized_image.reshape(target_shape)
        return resized_image

    def draw_detections(frame, frame_processor, detections, output_transform):
        size = frame.shape[:2]
        frame = output_transform.resize(frame)
        coordinates=[]
        jout='{"object":" ","Latency":" ","FPS":" "}'
        y=json.loads(jout)
        for roi in detections:

            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), size[1])
            ymax = min(int(roi.position[1] + roi.size[1]), size[0])
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),(0,0,255), 2)
            coordinates.append("{"+"'name':"+","+"'face'"+","+"'coordinates':"+('{:4},{:4},{:4},{:4} '.format(xmin, ymin, xmax, ymax))+"}")
            #print(coordinates)
        return frame,coordinates

    def center_crop(frame, crop_size):
        fh, fw, _ = frame.shape
        crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                    (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                    :]

    def face_detect(self,model_path: str,VEDIO_PATH: str,DEVICE: str):
        try:
            self.m_fd=model_path
            self.IN=VEDIO_PATH
            self.d_fd=DEVICE

            cap = open_images_capture(self.IN,self.loop)
            frame_processor = FrameProcessor(self.m_fd,self.fd_input_size,self.t_fd,self.exp_r_fd,self.d_fd,self.allow_grow,self.no_show)

            frame_num = 0
            metrics = PerformanceMetrics()
            presenter = None
            output_transform = None
            input_crop = None
            if self.crop_size[0] > 0 and self.crop_size[1] > 0:
                input_crop = np.array(self.crop_size)
            elif not (self.crop_size[0] == 0 and self.crop_size[1] == 0):
                raise ValueError('Both crop height and width should be positive')
            video_writer = cv2.VideoWriter()

            while True:
                start_time = perf_counter()
                frame = cap.read()
                if frame is None:
                    if frame_num == 0:
                        raise ValueError("Can't read an image from the input")
                    break
                if input_crop is not None:
                    frame = center_crop(frame, input_crop)
                if frame_num == 0:
                    output_transform = OutputTransform(frame.shape[:2], self.output_resolution)
                    if self.output_resolution:
                        output_resolution = output_transform.new_resolution
                    else:
                        output_resolution = (frame.shape[1], frame.shape[0])
                    presenter = monitors.Presenter(self.utilization_monitors, 55,
                                                (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                    if self.output and not video_writer.open(self.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                            cap.fps(), output_resolution):
                        raise RuntimeError("Can't open video writer")

                detections = frame_processor.process(frame)
                presenter.drawGraphs(frame)
                frame,coordinates= Face_Detector.draw_detections(frame, frame_processor, detections, output_transform)
                metrics.update(start_time, frame)
                total_latency, total_fps = metrics.get_total()
                Latency=str("{:.1f} ms".format(total_latency * 1e3))
                FPS=str("{:.1f}".format(total_fps))
                #metrics.log_total()
                jout='{"object":" ","Latency":" ","FPS":" "}'
                y=json.loads(jout)
                y["object"]=coordinates
                y["Latency"]=Latency
                y["FPS"]=FPS
                print(y)
                frame_num += 1
                if video_writer.isOpened() and (self.output_limit <= 0 or frame_num <= self.output_limit):
                    video_writer.write(frame)

                if not self.no_show:
                    cv2.imshow('Face recognition demo', frame)
                    key = cv2.waitKey(1)
                    # Quit
                    if key in {ord('q'), ord('Q'), 27}:
                        break
                    presenter.handleKey(key)
            for rep in presenter.reportMeans():
                log.info(rep)
            Stat='{"status": 0, "error":"None "}'
            Status=json.loads(Stat)
            return y,Status
        except Exception as e:
            out = '{"object":" ","confidence":" ","coordinates":" ","Latency":" ","FPS":" "}'
            Stat='{"status": 1, "error":" "}'
            output=json.loads(out)
            Status=json.loads(Stat)
            Status["error"]=e
            return output,Status



